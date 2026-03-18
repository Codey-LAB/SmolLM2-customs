---
title: SmolLM2 Customs
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
short_description: Showcase — Build your own free LLM service and plug it into any hub
---

# SmolLM2 Customs — Build Your Own LLM Service

> A showcase: how to build a free, private, OpenAI-compatible LLM service on HuggingFace Spaces and plug it into any hub or application — no GPU, no money, no drama.

> [!IMPORTANT]
> This project is under active development — always use the latest release from [Codey Lab](https://github.com/Codey-LAB/SmolLM2-customs) *(more stable builds land there first)*.
> This repo ([DEV-STATUS](https://github.com/VolkanSah/SmolLM2-custom)) is where the chaos happens. 🔬 A ⭐ on the repos would be cool 😙

---

## What is this?

A minimal but production-ready LLM service built on:

- **SmolLM2-360M-Instruct** — 269MB, Apache 2.0, runs on 2 CPUs and ~2GB RAM for free
- **FastAPI** — OpenAI-compatible `/v1/chat/completions` endpoint
- **ADI** (Anti-Dump Index) — filters low-quality requests before AND after inference
- **HF Dataset** — logs every request automatically for later finetuning
- **GitHub Actions** — one-click remote training pipeline, no SSH required

The point is not the model — the point is the pattern. Fork it, swap SmolLM2 for any model you want, and you have your own private LLM API running for free.

---

## The Real Use Case: AI for Broke Admins 💀

This isn't built to replace ChatGPT or to impress investors.

It's built because after years of Linux/security work you accumulate ~1984 commands in your head — and keeping them all sharp gets harder over time. The goal is a **small, private, terminal-drilled assistant** that:

- Knows your exact workflows — not generic tutorials
- Runs on a 2010 ThinkPad X201 with 4 CPUs and 8GB RAM
- Works fully offline when needed
- Doesn't explain things you already know — just outputs the command
- Gets smarter the more you use it, because every request feeds the training loop

Not a chatbot. Not an AI agent. A **drilled assistant** — trained on your own work, your own tools, your own patterns.

```
You:  "local pcap capture"
ShellMaster: tcpdump -i eth0 -w capture.pcap
             tshark -i eth0 -w capture.pcap

You:  "nmap stealth syn scan"
ShellMaster: nmap -sS 192.168.1.1

You:  "c2 jitter detection wireshark"
ShellMaster: dns && frame.time_delta
             frame.time_delta.stddev < 0.05 → automated behavior
```

No preamble. No explanation. Just the tool.

---

## How it works

```
Request
    ↓
ADI Score (input quality check)
    ↓
REJECT        → returns improvement suggestions, logs to dataset
MEDIUM/HIGH   → SmolLM2 answers, ADI scores the output too
SmolLM2 fails → 503 → hub fallback chain kicks in (gemini, openrouter, ...)
    ↓
Response logged to private HF Dataset (Parquet)
    ↓
You trigger GitHub Actions when ready → training pipeline runs
    ↓
Finetuned weights pushed to private model repo
    ↓
Next Space restart loads your model automatically
```

---

## The Training Loop — GitHub Actions

This is the actual clue of the whole setup. Every interaction gets logged. When you have enough data, one click in GitHub Actions kicks off the full pipeline — no SSH, no HF UI, no manual work:

```
[GitHub Actions — workflow_dispatch]
        │
        ├── mode: export
        │     └── dumps HF dataset → /tmp/train_data.jsonl
        │         (filtered: REJECT entries skipped, BLOCKED included)
        │         server keeps running, data waits in /tmp/
        │
        ├── mode: validate          ← optional
        │     └── checks ADI weight accuracy against labeled data
        │         useful if you manually edited the CSV via Parquet-Sync
        │
        ├── mode: finetune
        │     └── TRL SFTTrainer on CPU (batch=1, grad_accum=4)
        │         → saves to /tmp/finetuned_model
        │         → pushes weights to your private HF model repo
        │         → updates model card automatically
        │
        └── mode: all / export_validate
              └── runs steps sequentially in one pipeline run
```

**The workflow is safe by design:**
- `workflow_dispatch` only — only you can trigger it (Write access required)
- Concurrency guard — no parallel training runs, ever
- Retry with exponential backoff — HF Spaces wake up slowly sometimes
- Dry-run mode — prints curl commands without executing, for testing
- 4xx → immediate abort, no pointless retries on auth errors
- Full pipeline summary in GitHub Actions UI after every run

```yaml
# .github/workflows/smollm2-training.yml
on:
  workflow_dispatch:
    inputs:
      mode:
        type: choice
        options: [all, export, validate, finetune, export_validate]
      dry_run:
        type: boolean
```

**The typical workflow:**

```
1. Use the service → requests get logged automatically
2. Check your HF dataset → review what was logged
3. Edit CSV if needed via Parquet-Sync (fix responses, add labels)
4. GitHub Actions → export → (validate) → finetune
5. Space restarts → loads your finetuned model
6. Repeat
```

Or skip to `finetune` directly if you already exported and edited manually — the pipeline is flexible.

---

## Endpoints

```
GET  /                        → status
GET  /v1/health               → health check (auth required)
POST /v1/chat/completions     → OpenAI-compatible inference
POST /v1/train/execute?mode=  → remote training trigger
                                 mode: export | validate | finetune
```

---

## Plug into any Hub

Works out of the box with [Multi-LLM-API-Gateway](https://github.com/VolkanSah/Multi-LLM-API-Gateway):

```ini
[LLM_PROVIDER.smollm]
active        = "true"
base_url      = "https://YOUR-USERNAME-smollm2-customs.hf.space/v1"
env_key       = "SMOLLM_API_KEY"
default_model = "smollm2-360m"
models        = "smollm2-360m, YOUR-USERNAME/your-finetuned-model"
fallback_to   = "gemini"
[LLM_PROVIDER.smollm_END]
```

Any OpenAI-compatible client works the same way.

---

## Secrets (HF Space Settings)

| Secret | Required | Description |
|--------|----------|-------------|
| `SMOLLM_API_KEY` | recommended | Locks the endpoint — set same value in your hub and GitHub Secret |
| `HF_TOKEN` or `TEST_TOKEN` | optional | HF auth for dataset + model repo access |
| `MODEL_REPO` | optional | Base model override (default: `HuggingFaceTB/SmolLM2-360M-Instruct`) |
| `DATASET_REPO` | optional | Your private HF dataset for logging |
| `PRIVATE_MODEL_REPO` | optional | Your private model repo for finetuned weights |

**GitHub Secret needed for Actions:**
```
Settings → Secrets → Actions → New repository secret
Name: SMOLLM_API_KEY
Value: same as your HF Space secret
```

**Auth modes:**
```
SMOLLM_API_KEY not set  → open access (demo/showcase mode)
SMOLLM_API_KEY set      → protected (production mode)
Space private           → double protection (HF gate + your key)
```

---

## ADI Routing

ADI runs **twice** — on input and on output. Low-quality requests never reach the model. Low-quality responses never reach the user.

| Decision | Action |
|----------|--------|
| `HIGH_PRIORITY` | SmolLM2 handles it |
| `MEDIUM_PRIORITY` | SmolLM2 handles it |
| `REJECT` | Returns improvement suggestions, logs to dataset |
| `BLOCKED` | Hard block — dangerous/unauthorized patterns |
| SmolLM2 fails | 503 → hub fallback chain |

---

## Hardware Reality

```
HF Space (free tier):   2 CPU  /  2-3 GB RAM  → inference + training
ThinkPad X201 (2010):   4 CPU  /  8 GB RAM    → local deployment
Neon PostgreSQL (free): 4 CPU  / 22 GB RAM    → Guardian layer DB
```

The database is heavier than the model. SmolLM2-360M fits in 2GB. That's the point.

---

## Stack

| Component | What it does |
|-----------|-------------|
| `main.py` | FastAPI, auth, routing, training trigger endpoint |
| `smollm.py` | Inference engine, lazy loading |
| `model.py` | HF token resolution, dataset + model repo access |
| `adi.py` | Request quality scoring (input + output) |
| `train.py` | Dataset export, ADI validation, TRL SFTTrainer finetuning |
train_local.sh not ready

---

## Part of

| Project | Role |
|---------|------|
| [Multi-LLM-API-Gateway](https://github.com/VolkanSah/Multi-LLM-API-Gateway) | The hub — Guardian pattern, MCP, fallback chain |
| [Anti-Dump-Index](https://github.com/VolkanSah/Anti-Dump-Index) | ADI algorithm |
| [Parquet-Sync](https://github.com/VolkanSah/Parquet-Sync) | Edit HF dataset Parquet files as human-readable CSV |
| [Blue Earth](https://github.com/VolkanSah/Hack-the-Planet) | The security workflow the ShellMaster is drilled on |

---

## License

Dual-licensed:

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Ethical Security Operations License v1.1 (ESOL)](ESOL) — mandatory, non-severable

By using this software you agree to all ethical constraints defined in ESOL v1.1.

---

*Built with Claude (Anthropic) as a typing assistant — tabs, docs, and the occasional bug fix.*
*Architecture, security decisions, and the 1984 commands: Volkan Kücükbudak.* 😄

