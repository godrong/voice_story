#!/usr/bin/env python3
"""CosyVoice 3 LoRA fine-tuning — complete self-contained training script.

Upload and run:
  scp lora_train.py root@host:/tmp/
  ssh root@host "cd /root/CosyVoice && python /tmp/lora_train.py --lora_rank 16"

Features:
  - Manual LoRA injection into CosyVoice3LM Qwen2 attention layers
  - Text-format data pipeline (replaces parquet_opener)
  - Mixed precision (AMP) on single GPU
  - Periodic checkpoint saving
"""

import sys, os, argparse, json, logging, io, math
from functools import partial

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 0. Path setup
# ---------------------------------------------------------------------------
COSYVOICE = os.environ.get("COSYVOICE_ROOT", "/root/CosyVoice")
sys.path.insert(0, COSYVOICE)
sys.path.insert(0, os.path.join(COSYVOICE, "third_party", "Matcha-TTS"))

from hyperpyyaml import load_hyperpyyaml
from cosyvoice.dataset.dataset import DataList, Processor, read_lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lora_train")


# ---------------------------------------------------------------------------
# 1. LoRA module (manual, no peft dependency)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear + LoRA: out = Wx + (alpha/r) * B(A(dropout(x)))"""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        self.lora_dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + self.scaling * ((self.lora_dropout(x) @ self.lora_A) @ self.lora_B)


def inject_lora(model, rank=16, alpha=32, dropout=0.05):
    """Inject LoRALinear into all Qwen2 attention projection layers inside model.llm."""
    target = model.llm
    injected = []
    for name, mod in list(target.named_modules()):
        if isinstance(mod, nn.Linear) and any(k in name for k in
                                               ["q_proj", "k_proj", "v_proj", "o_proj"]):
            parent = target
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout))
            injected.append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, len(injected), trainable, total


def save_lora(model, path):
    os.makedirs(path, exist_ok=True)
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            state[name] = param.data.clone().cpu()
    torch.save(state, os.path.join(path, "lora_weights.pt"))
    logger.info("Saved LoRA weights to %s (%d params)", path, len(state))


# ---------------------------------------------------------------------------
# 2. Text-format data opener (replaces parquet_opener)
# ---------------------------------------------------------------------------

def text_opener(data, mode="train", tts_data={}):
    """Parse tab-separated lines: utt_id<TAB>wav_path<TAB>text.
    Reads wav files into audio_data bytes, matching parquet opener format."""
    for sample in data:
        line = sample.get("src", "")
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        utt_id, wav_path, text = parts[0], parts[1], "\t".join(parts[2:])
        try:
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
        except Exception:
            continue
        out = {"utt": utt_id, "wav": wav_path, "text": text,
               "audio_data": wav_bytes, "sample_rate": 16000}
        if mode != "train":
            out.update({"tts_index": 0, "tts_text": text})
        yield out


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda")
    logger.info("Device: %s", device)
    logger.info("VRAM free: %.0f MB", torch.cuda.get_device_properties(0).total_memory / 1e6)

    # Load model from config
    config_path = os.path.join(args.model_dir, "cosyvoice3.yaml")
    qwen_path = os.path.join(args.model_dir, "CosyVoice-BlankEN")
    checkpoint = os.path.join(args.model_dir, "llm.pt")

    with open(config_path) as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": qwen_path})

    model = configs["llm"]
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    del state_dict
    logger.info("Model loaded: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    # Inject LoRA
    model, n_layers, trainable, total = inject_lora(
        model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    logger.info("LoRA rank=%d: %d layers, trainable %.1fM / %.1fM (%.2f%%)",
                args.lora_rank, n_layers, trainable / 1e6, total / 1e6, 100 * trainable / total)

    # Optimizer
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps,
                                                        eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Data pipeline (config pipeline with text_opener replacing parquet_opener)
    data_list = read_lists(args.train_data)
    dataset = DataList(data_list, shuffle=True, partition=False)
    pipeline = list(configs["data_pipeline"])
    pipeline[0] = text_opener  # swap parquet → text
    for func in pipeline:
        dataset = Processor(dataset, func, mode="train")

    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    losses, step = [], 0
    logger.info("Training: max_steps=%d, lr=%.2e, batch_size=dynamic, rank=%d",
                args.max_steps, args.lr, args.lora_rank)
    model.train()

    for epoch in range(args.epochs):
        batch_count = 0
        for batch in loader:
            if step >= args.max_steps:
                break
            try:
                loss = model.forward(batch, device)["loss"]
            except Exception as e:
                if batch_count == 0:
                    logger.error("Forward failed: %s", str(e)[:100])
                continue

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            lv = loss.item()
            losses.append(lv)
            step += 1
            batch_count += 1

            if step % args.log_interval == 0:
                avg = sum(losses[-args.log_interval:]) / args.log_interval
                logger.info("step %4d/%d | loss=%7.4f | lr=%.2e | epoch=%d/%d",
                            step, args.max_steps, avg, sched.get_last_lr()[0],
                            epoch, args.epochs)

            if step % args.save_interval == 0:
                save_lora(model, os.path.join(args.output_dir, "step_%d" % step))

        logger.info("Epoch %d complete: %d batches, avg_loss=%.4f",
                    epoch, batch_count,
                    sum(losses[-batch_count:]) / max(batch_count, 1) if batch_count else 0)
        if step >= args.max_steps:
            break

    # Final save
    save_lora(model, os.path.join(args.output_dir, "final"))
    metrics = {"lora_rank": args.lora_rank, "steps": step, "n_layers": n_layers,
               "trainable_M": round(trainable / 1e6, 1),
               "total_M": round(total / 1e6, 1),
               "final_loss": sum(losses[-100:]) / max(len(losses), 1)}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Done! %s", json.dumps(metrics))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CosyVoice 3 LoRA Fine-tuning")
    p.add_argument("--model_dir", default=os.path.join(COSYVOICE, "pretrained_models",
                   "Fun-CosyVoice3-0.5B"))
    p.add_argument("--train_data", required=True, help="data.list file (tab-separated)")
    p.add_argument("--output_dir", default="/root/autodl-tmp/exp003_lora")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=50)
    train(p.parse_args())
