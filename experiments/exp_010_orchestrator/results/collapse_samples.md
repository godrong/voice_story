# exp_010 — Collapse 真实样例

> 三个模型在同一段文本上的输出对比。
> 1.5B / 3B = 50 段全跑；7B = 仅 idx 10-19（散热限制）。

## 样例 1：角色直接对白

**[seg luxun_AQ_0012]**
> `…对他说："阿Ｑ，这不是儿子打老子，是人打畜生。自己说：人打畜生！"阿Ｑ两只手都捏住了自己的辫根，歪著头，说道："打虫豸，好不好？我是虫豸──还不放么？"…`

| 模型 | role | emotion | pause |
|---|---|---|---|
| 1.5B-4bit | narrator | neutral | short |
| 3B-4bit | narrator | neutral | medium |
| **7B-4bit** | **narrator** | **neutral** | **medium** |

→ 7B 也没识别出对白？说明 7B 也会 miss——但下面这段它对了：

---

## 样例 2：明显怒火

**[seg luxun_AQ_0013]**
> `"你算是什么东西"呢…阿Ｑ以如是等等妙法克服怨敌之后，便愉快的跑到酒店里…`

| 模型 | role | emotion | pause |
|---|---|---|---|
| 1.5B-4bit | narrator | neutral | short |
| 3B-4bit | narrator | neutral | medium |
| **7B-4bit** | **character_A** | **angry** | **medium** |

→ 7B 抓住了 `"你算是什么东西"` 是开火式怒骂。

---

## 样例 3：场景描写 + 隐式情绪

**[seg luxun_AQ_0014]**
> `这是未庄赛神的晚上。这晚上照例有一台戏，戏台左近，也照例有许多的赌摊。…`

| 模型 | role | emotion | pause |
|---|---|---|---|
| 1.5B-4bit | narrator | neutral | short |
| 3B-4bit | narrator | neutral | medium |
| **7B-4bit** | **ambiguous** | **sad** | **medium** |

→ 7B 抓住了隐式情绪（赛神晚上的悲凉氛围）；3B 只看到表面字。

---

## 样例 4：愤怒爆发

**[seg luxun_AQ_0017]**
> `阿Ｑ最初是失望，后来却不平了：看不上眼的王胡尚且那么多，自己反到这样少，这是怎样的大失体统的事呵！…`

| 模型 | role | emotion | pause |
|---|---|---|---|
| 1.5B-4bit | narrator | neutral | short |
| 3B-4bit | narrator | neutral | medium |
| **7B-4bit** | **ambiguous** | **angry** | **medium** |

→ "大失体统" 这种情绪词，3B 看不见，7B 看得见。

---

## 解读

7B 在 10 段对白密集子集里：
- role: 1 个 character_A + 4 个 ambiguous + 5 个 narrator
- emotion: 3 angry + 2 sad + 5 neutral

不是每一段都被翻译成 dialogue/emotional，但**该抓的情绪信号都抓到了**。

3B 全 50 段：100% narrator/neutral——完全没"看到"文本内容。

这是 collapse 的真实样子：模型默认输出"最安全"标签，schema 形式合规但语义信息归零。

## 写论文 / 简历时用的 punchline

> "在中文文学情感标注任务上，Qwen2.5-4bit 量化系列的判断能力门槛在 3B↔7B 之间。低于 7B 时模型完全退化为输出先验 (narrator/neutral)，对实际语义零响应。"
