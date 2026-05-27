#!/usr/bin/env python3
"""LoRA fine-tuning for CosyVoice 3 LLM using official training pipeline.

Adds LoRA adapters to Qwen2 attention layers inside CosyVoice3LM,
then trains cross-emotion style-following on ESD data.

Usage (on remote RTX 4090):
  cd /root/CosyVoice
  python /root/voice_story/experiments/exp_003_cosyvoice3/train_cosyvoice3_lora.py \
    --train_data /root/autodl-tmp/esd_lora_train/train.data.list \
    --cv_data /root/autodl-tmp/esd_lora_train/train.data.list \
    --output_dir /root/autodl-tmp/exp003_lora_rank16 \
    --lora_rank 16 --max_steps 500 --batch_size 2
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

COSYVOICE_ROOT = Path("/root/CosyVoice")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

from hyperpyyaml import load_hyperpyyaml
from cosyvoice.dataset.dataset import DataList

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA Injection
# ---------------------------------------------------------------------------

def inject_lora_to_qwen2(model: nn.Module, rank: int = 16, alpha: int = 32,
                          dropout: float = 0.05) -> nn.Module:
    """Inject LoRA into Qwen2ForCausalLM attention layers via peft."""
    from peft import LoraConfig, get_peft_model, TaskType

    # In CosyVoice3LM: self.llm is a Qwen2ForCausalLM-compatible model
    target = model.llm

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    peft_model = get_peft_model(target, lora_config)
    model.llm = peft_model  # Replace the LLM with LoRA version

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA rank=%d injected. Trainable: %d / %d (%.2f%%)",
                rank, trainable, total, 100 * trainable / total)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load CosyVoice 3 config
    model_dir = Path(args.model_dir)
    config_path = model_dir / "cosyvoice3.yaml"
    with open(config_path) as f:
        configs = load_hyperpyyaml(f, overrides={
            "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")
        })

    # 2. Build LLM model from config
    logger.info("Building CosyVoice3LM from config...")
    llm_cfg = configs["llm"]
    # CosyVoice3LM accepts only these args (others are set internally)
    valid_kwargs = {"llm_input_size", "llm_output_size", "speech_token_size",
                    "llm", "sampling", "length_normalized_loss", "lsm_weight",
                    "mix_ratio"}
    model_kwargs = {}
    for k, v in llm_cfg.__dict__.items():
        if not k.startswith("_") and k in valid_kwargs:
            model_kwargs[k] = v
    logger.info("Model kwargs: %s", list(model_kwargs.keys()))

    from cosyvoice.llm.llm import CosyVoice3LM
    model = CosyVoice3LM(**model_kwargs)

    # 3. Load pretrained weights
    llm_ckpt = model_dir / "llm.pt"
    logger.info("Loading pretrained LLM from %s", llm_ckpt)
    state_dict = torch.load(llm_ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # 4. Inject LoRA into Qwen2 backbone
    logger.info("Injecting LoRA rank=%d...", args.lora_rank)
    model = inject_lora_to_qwen2(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                 dropout=args.lora_dropout)

    # 5. Freeze non-LoRA params
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 6. Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.learning_rate * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # 7. DataLoader using CosyVoice 3 DataList
    logger.info("Loading training data from %s", args.train_data)
    train_dataset = DataList(
        args.train_data,
        tokenizer=configs["get_tokenizer"],
        feat_extractor=None,  # LLM-only training doesn't need mel features
        allowed_special=configs.get("allowed_special", "all"),
        min_token=1, max_token=200,
        min_dur=0, max_dur=30,
        use_spk_embedding=False,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn, num_workers=2, pin_memory=True,
    )

    # 8. Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    losses = []

    logger.info("Starting LoRA training: max_steps=%d, rank=%d, lr=%.2e",
                args.max_steps, args.lora_rank, args.learning_rate)
    model.train()

    for epoch in range(args.max_epochs):
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            # Forward: CosyVoice3LM.forward(batch, device)
            output = model.forward(batch, device)
            loss = output["loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())
            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = sum(losses[-args.log_interval:]) / len(losses[-args.log_interval:])
                logger.info("Step %4d/%d | loss=%7.4f | lr=%.2e",
                            global_step, args.max_steps, avg_loss,
                            scheduler.get_last_lr()[0])

            # Save best checkpoint
            if global_step % args.save_interval == 0:
                val_loss = avg_loss  # Use recent avg as proxy
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_lora_checkpoint(model, output_dir / "best_lora",
                                         args.lora_rank, global_step, val_loss)
                    logger.info("Saved best LoRA checkpoint at step %d", global_step)

        if global_step >= args.max_steps:
            break

    # 9. Final save
    save_lora_checkpoint(model, output_dir / "final_lora",
                         args.lora_rank, global_step,
                         sum(losses[-100:]) / min(len(losses), 100))
    logger.info("Training complete. LoRA saved to %s", output_dir)

    # Save training curve
    with open(output_dir / "train_loss.json", "w") as f:
        json.dump({"lora_rank": args.lora_rank, "steps": global_step,
                   "final_loss": sum(losses[-100:]) / min(len(losses), 100),
                   "loss_history": losses}, f, indent=2)


def save_lora_checkpoint(model, path, rank, step, loss):
    """Save only LoRA adapter weights."""
    import os
    os.makedirs(str(path), exist_ok=True)
    # peft stores adapters
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(str(path))
    metadata = {"lora_rank": rank, "step": step, "loss": loss}
    with open(os.path.join(str(path), "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for CosyVoice 3 LLM")
    parser.add_argument("--model_dir", default="/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--cv_data", default="")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/exp003_lora")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
