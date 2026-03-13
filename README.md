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

- **SmolLM2-360M-Instruct** — 269MB, Apache 2.0, runs on 2 CPUs for free
- **FastAPI** — OpenAI-compatible `/v1/chat/completions` endpoint
- **ADI** (Anti-Dump Index) — filters low-quality requests before they hit the model
- **HF Dataset** — logs every request for later analysis and finetuning

The point is not the model — the point is the pattern. Fork it, swap SmolLM2 for any model you want, and you have your own private LLM API running for free.

---

## How it works

```
Request
    ↓
ADI Score (is this request worth answering?)
    ↓
REJECT        → returns improvement suggestions, logs to dataset
MEDIUM/HIGH   → SmolLM2 answers, logs to dataset
SmolLM2 fails → returns 503 → hub fallback chain kicks in
```

---

## Endpoints

```
GET  /                       → status
GET  /v1/health              → health check
POST /v1/chat/completions    → OpenAI-compatible inference
```

---

## Plug into any Hub (one config block)

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
| `SMOLLM_API_KEY` | recommended | Locks the endpoint — set same value in your hub |
| `HF_TOKEN` or `TEST_TOKEN` | optional | HF auth for dataset + model repo access |
| `MODEL_REPO` | optional | Base model override (default: `HuggingFaceTB/SmolLM2-360M-Instruct`) |
| `DATASET_REPO` | optional | Your private HF dataset for logging |
| `PRIVATE_MODEL_REPO` | optional | Your private model repo for finetuned weights |

**Auth modes:**
```
SMOLLM_API_KEY not set  → open access (demo/showcase mode)
SMOLLM_API_KEY set      → protected (production mode)
Space private           → double protection (HF gate + your key)
```

---

## ADI Routing

| Decision | Action |
|----------|--------|
| `HIGH_PRIORITY` | SmolLM2 handles it |
| `MEDIUM_PRIORITY` | SmolLM2 handles it |
| `REJECT` | Returns suggestions, logs to dataset |
| SmolLM2 fails | 503 → hub fallback chain |

---

## Training Utilities

Every request is logged to your private HF dataset. Use it to improve over time:

```bash
python train.py --mode export    # export dataset → JSONL
python train.py --mode validate  # validate ADI weights against labeled data
python train.py --mode finetune  # finetune SmolLM2 on your data (coming soon)
```

Once you have enough data → finetune → push to your private model repo → Space loads it automatically next restart.

---

## Stack

| Component | What it does |
|-----------|-------------|
| `main.py` | FastAPI, auth, routing |
| `smollm.py` | Inference engine, lazy loading |
| `model.py` | HF token resolution, dataset + model repo access |
| `adi.py` | Request quality scoring |
| `train.py` | Dataset export, ADI validation, finetuning |

---

## Part of

- [Multi-LLM-API-Gateway](https://github.com/VolkanSah/Multi-LLM-API-Gateway) — the hub this was built for
- [Anti-Dump-Index](https://github.com/VolkanSah/Anti-Dump-Index) — the ADI algorithm
