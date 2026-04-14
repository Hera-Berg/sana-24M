# Sana 🪼 — Assistant

A ~24M parameter language model trained from scratch. No Hugging Face transformers. No SentencePiece. No flash-attn. Every component hand-implemented in pure PyTorch.

---

## Architecture

| Component         | Detail                                         |
| ----------------- | ---------------------------------------------- |
| Parameters        | ~24M unique (lm_head weight tied to embedding) |
| Hidden dim        | 384                                            |
| Layers            | 10 TransformerBlocks                           |
| Attention heads   | 6                                              |
| Head dim          | 64 (384 / 6)                                   |
| FFN intermediate  | 1,024 (384 × 2.667 rounded, SwiGLU)            |
| Vocabulary        | 16,000 tokens (BPE, pure Python)               |
| Max sequence      | 512 tokens                                     |
| Position encoding | RoPE (Rotary Position Embeddings), base=10000  |
| Normalisation     | RMSNorm (pre-norm architecture)                |
| Activation        | SwiGLU (gated FFN)                             |
| KV Cache          | Yes (per-layer, allocated during generation)   |
| Tied embeddings   | Yes (input/output share one weight matrix)     |
| Dropout           | 0.1 during pretrain, 0.0 during SFT            |

---

## Special Tokens

| ID  | Token            | Purpose                   |
| --- | ---------------- | ------------------------- |
| 0   | `<\|pad\|>`      | Padding                   |
| 1   | `<\|user\|>`     | User turn marker          |
| 2   | `<\|sana\|>`     | Sana turn marker          |
| 3   | `<\|end\|>`      | End of turn / EOS         |
| 4   | `<tool>`         | Tool call open            |
| 5   | `</tool>`        | Tool call close           |
| 6   | `<tool_result>`  | Tool result open          |
| 7   | `</tool_result>` | Tool result close         |
| 8   | `<sana_salute>`  | Sana emotion: 🪼 greeting |
| 9   | `<sana_happy>`   | Sana emotion: ✨ happy    |
| 10  | `<sana_think>`   | Sana emotion: 🤔 thinking |
| 11  | `<sana_sad>`     | Sana emotion: 😔 sad      |
| 12  | `<sana_smug>`    | Sana emotion: 😏 smug     |

BPE tokens start at ID 13. The word "Sana" is hardcoded as ID 13 to guarantee single-token encoding regardless of BPE merges.

---

## Design Philosophy

**SFT teaches format and persona, not facts.** At 24M parameters, the model cannot reliably memorise specific factual content — attempting to do so via SFT overwrites pretrain knowledge and causes hallucination. Instead:

- **Pretraining** on ~2B tokens of FineWeb-Edu gives the model broad factual knowledge
- **SFT** teaches _how_ to respond: emotion tokens, short answers, dry personality, empathy
- **Factual accuracy** at inference time is the job of RAG (retrieval-augmented generation), which is stubbed out in `inference/inference.py` and not yet implemented for the 24M model

The SFT dataset is therefore split into three equal behavioral categories:

| Category   | What it teaches                                                         |
| ---------- | ----------------------------------------------------------------------- |
| `identity` | Who Sana is, greetings, goodbyes, opinions, personality                 |
| `factual`  | Format of factual responses — short, topic-first, correct emotion token |
| `empathy`  | Emotional mirroring — sad user → `<sana_sad>`, happy → `<sana_happy>`   |

---

## The Bioluminescence Experiment 🔬

**What it proves:** Knowledge from pretraining persists through SFT even when the topic never appears in the SFT dataset.

**Setup:** "bioluminescence" does not appear anywhere in `data/sft_direct.jsonl`. It does appear in FineWeb-Edu pretraining data.

**Test:** Ask Sana `"What is bioluminescence?"` after fine-tuning. Despite never seeing the word in SFT, Sana should be able to describe it from pretrain knowledge.

```bash
# Confirm word is absent from SFT data
grep -i "bioluminescen" data/sft_direct.jsonl && echo "FOUND" || echo "CLEAN"

# Ask the model
python inference/inference.py --checkpoint checkpoints/finetune
# → What is bioluminescence?
```

This demonstrates that SFT shapes _how_ a model responds without erasing what it knows.

---

## Full Pipeline

### 0. Install dependencies

```bash
pip install torch pyyaml datasets numpy tqdm openai
```

---

### 1. Download pretraining data

Downloads ~2B tokens from FineWeb-Edu (HuggingFace). Resume-safe — re-running skips completed files.

```bash
python pretrain/download_data.py \
    --output_dir data/pretrain_raw \
    --max_tokens 1200
```

- Runtime: ~4–8h depending on internet speed
- Output: `data/pretrain_raw/fineweb_00000.txt` … `fineweb_NNNNN.txt`
- ~1,000 files, 2,000 articles each, ~20GB total
- Unicode normalization applied at write time (curly quotes, em dashes → ASCII)

---

### 2. Train BPE tokenizer

Trains a 16,000-token BPE vocabulary on a sample of the pretrain data, with SFT dialogue lines mixed in to ensure conversational tokens merge cleanly.

```bash
python tokenizer/train_tokenizer.py \
    --data_dir data/pretrain_raw \
    --vocab_size 16000 \
    --sample_lines 100000 \
    --output tokenizer/sana_tokenizer/tokenizer.json \
    --force
```

- Runtime: ~20–40 minutes
- The `--dialogue_dir` flag ensures words like "Sana", "hi", "jellyfish" become single tokens
- Output: `tokenizer/sana_tokenizer/tokenizer.json`

---

### 3. Tokenize pretraining data

Pre-tokenizes all `.txt` files into binary uint16 shards. Zero tokenization overhead at training time.

```bash
python pretrain/tokenize_data.py \
    --data_dir data/pretrain_raw \
    --output_dir data/pretrain_bin \
    --shard_size 200 \
    --workers 32
```

- Runtime: ~30–60 minutes
- Output: `data/pretrain_bin/shard_00000.bin`, `shard_00000.json`, ..., `meta.json`
- Each shard is ~200MB, ~100M tokens

---

### 4. Pretrain

**RTX 4060 (8GB):**

```bash
python pretrain/pretrain.py --config pretrain/config.yaml
```

**H100 (80GB):**

```bash
python pretrain/pretrain.py --config pretrain/config_h100.yaml
```

**Resume from checkpoint:**

```bash
python pretrain/pretrain.py --config pretrain/config.yaml \
    --resume checkpoints/pretrain/ckpt_015000.pt
```

- RTX 4060 runtime: ~4–5h (batch=16, accum=4, ~30,000 steps, ~1B tokens)
- H100 runtime: ~10–15min (batch=256, no accum, full 2B tokens)
- Checkpoints saved every 1,000 steps to `checkpoints/pretrain/`

**Expected loss curve (RTX 4060, 24M model):**

| Step   | Loss |
| ------ | ---- |
| 100    | ~4.7 |
| 500    | ~3.3 |
| 2,000  | ~2.6 |
| 10,000 | ~2.3 |
| 15.000 | ~2.2 |

---

### 5. Test pretrained model

```bash
python test_pretrain.py --checkpoint checkpoints/pretrain
```

- 25 prompts across 5 categories: coherence, general science, biology, ocean/jellyfish, sentence continuation
- Pass threshold: 60%
- Output includes ASCII bar chart per category

---

### 6. Generate SFT data

Generates ~1,200 training examples using GPT-4o-mini. Requires an OpenAI API key. Costs approximately $0.10–0.20.

```bash
export OPENAI_API_KEY="your-key-here"

python pretrain/generate_direct_sft.py \
    --output data/sft_direct.jsonl \
    --n 1500 \
    --workers 30
```

- Generates 1,500 examples for surplus, then subsamples to ~1,200 with minimum token floors
- Three equal categories: `identity`, `factual`, `empathy`
- Each category: 1/3 single-turn, 1/3 continuation, 1/3 topic-switch (chaos)
- Runtime: ~3–5 minutes
- Validator enforces correct emotion tokens per category (wrong tokens are rejected and regenerated)

**Verify token distribution after generation:**

```bash
python3 -c "
import json
from collections import Counter
data = [json.loads(l) for l in open('data/sft_direct.jsonl') if l.strip()]
toks = Counter()
for d in data:
    for t in d['conversations']:
        if t['role'] == 'sana':
            c = t['content']
            for tok in ['<sana_salute>','<sana_happy>','<sana_think>','<sana_sad>','<sana_smug>']:
                if c.startswith(tok): toks[tok] += 1
total = sum(toks.values())
[print(f'  {t:<18} {c:>4}  ({c/total*100:.0f}%)') for t,c in sorted(toks.items(), key=lambda x:-x[1])]
print(f'  Total responses: {total}')
"
```

Target distribution: think ~25%, smug ~25%, sad ~20%, happy ~17%, salute ~13%.

---

### 7. Fine-tune

Update `finetune/config.yaml` with your pretrain checkpoint path, then:

```bash
python finetune/finetune.py --config finetune/config.yaml
```

- Runtime: ~5–10 minutes on RTX 4060
- Checkpoints saved per epoch to `checkpoints/finetune/`
- Default: 5 epochs, lr=1e-5, effective batch=16
- Loss is computed only on Sana's response tokens (user tokens masked to -100)

**Choosing the best checkpoint:** The real average loss per epoch is displayed/4 due to gradient accumulation. Target ~1.2–1.5 real avg loss. If step losses drop below 0.5 consistently, earlier checkpoint is better.

```bash
# Test a specific epoch checkpoint
python inference/inference.py \
    --checkpoint checkpoints/finetune/ckpt_epoch_03.pt \
    --temperature 0.4
```

---

### 8. Test fine-tuned model

```bash
python test_finetune.py --checkpoint checkpoints/finetune
```

- Single-turn: science, biology, jellyfish, math, history
- Multi-turn: science follow-ups, jellyfish life cycle, gravity + relativity
- Bioluminescence experiment: verifies pretrain knowledge retention
- Pass threshold: 70%

---

### 9. Interactive inference

```bash
python inference/inference.py --checkpoint checkpoints/finetune --temperature 0.4
```

Commands at the prompt:

- `reset` — clear conversation history
- `quit` / `exit` — exit

**Note on RAG:** `inference.py` contains a stubbed `RAGRetriever` class and `route_tool()` function. These are not implemented for the 24M model. Tool calls generated by the model will return `[Tool not implemented: ...]`. This is intentional — factual accuracy at 24M parameters comes from the pretrain data, not from runtime retrieval.

---

## SFT Data Format

Each line in `data/sft_direct.jsonl`:

```json
{
  "conversations": [
    { "role": "user", "content": "who are you" },
    {
      "role": "sana",
      "content": "<sana_smug> I'm Sana. Compact AI assistant, trained from scratch."
    }
  ],
  "_meta": {
    "category": "identity",
    "type": "single",
    "source": "generated"
  }
}
```

Multi-turn example:

```json
{
  "conversations": [
    { "role": "user", "content": "what do jellyfish eat" },
    {
      "role": "sana",
      "content": "<sana_think> Plankton, small fish, fish eggs. Passive hunters — drift and trap prey with tentacles."
    },
    { "role": "user", "content": "cool" },
    {
      "role": "sana",
      "content": "<sana_happy> Some species eat other jellyfish. A few can revert to juvenile form and essentially restart life."
    }
  ],
  "_meta": {
    "category": "factual",
    "type": "continuation",
    "source": "generated"
  }
}
```

**Rules:**

- Every Sana turn starts with exactly one emotion token
- Loss computed only on Sana turns (user tokens masked to -100)
- Max 35 words per response body (after emotion token)

---

## Repo Structure

```
sana/
├── model/
│   └── model.py                  ← RMSNorm, RoPE, SwiGLU, Attention, KVCache,
│                                    TransformerBlock, SanaModel, SanaConfig
├── tokenizer/
│   ├── train_tokenizer.py        ← Pure Python BPE training
│   ├── tokenizer.py              ← Pure Python tokenizer (json + re only)
│   └── sana_tokenizer/
│       └── tokenizer.json        ← (generated by train_tokenizer.py)
├── pretrain/
│   ├── download_data.py          ← FineWeb-Edu streaming, 2B token target
│   ├── tokenize_data.py          ← Pre-tokenize .txt → uint16 binary shards
│   ├── pretrain.py               ← Training loop (per-doc chunking, fp16/bf16)
│   ├── generate_direct_sft.py    ← SFT dataset generator (GPT-4o-mini, 3 categories)
│   ├── config.yaml               ← RTX 4060 pretrain config
│   └── config_h100.yaml          ← H100 pretrain config
├── finetune/
│   ├── finetune.py               ← SFT loop with response-token masking
│   └── config.yaml               ← SFT config (lr, epochs, checkpoint path)
├── inference/
│   └── inference.py              ← SanaInference class + CLI (RAG stubbed out)
├── test_pretrain.py              ← 25 prompts, 5 categories, ASCII bar chart
├── test_finetune.py              ← dialogue + bioluminescence + tool tests
└── README.md
```

---

## Key Implementation Notes

**Per-document chunking:** `TextDataset` processes each `.txt` file independently. Token arrays are never concatenated across files before chunking — cross-file chunking would train the model that EOS can appear mid-sentence.

**Unicode normalization:** `download_data.py` maps curly quotes, em dashes, ellipses, and common accented characters to ASCII equivalents before writing `.txt` files. `tokenize_data.py` applies the same pass as a safety net for files downloaded with older code. This prevents BPE byte-level fallback (`e2 80 99` → `\u2019`) from polluting the token stream.

**RoPE start_pos:** Generation passes `start_pos=current_length` to the attention layer at each decode step so cached positions are assigned correctly.

**Tied embeddings:** `self.lm_head.weight = self.embedding.weight` — one assignment, not two separate initialisations.

**No TemplateProcessing:** `encode("hello")` never contains `END_ID`. Verified by sanity check after tokenizer training.

**Loss masking:** Label IDs have `-100` on all user tokens and structural role-marker tokens. `F.cross_entropy(..., ignore_index=-100)` skips these positions.

**SFT signal:** With 1,200 samples and 5 epochs at lr=1e-5, signal ≈ 3.75e-3. Each behavioral pattern is seen ~20–25 times — enough for a 24M model to learn format without overwriting pretrain knowledge.
