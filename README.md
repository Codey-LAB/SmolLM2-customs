---
title: SmolLM2 Service
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
short_description: SmolLM2-360M OpenAI-compatible API with ADI routing
---

# SmolLM2 Service
##### only a stupid LLM assi for the hub runs on 2 CPUs for free :D

OpenAI-compatible LLM API powered by `SmolLM2-360M-Instruct` with integrated ADI (Anti-Dump Index) routing.

## Endpoints

```
GET  /v1/health              → status check
POST /v1/chat/completions    → OpenAI-compatible inference
```

## Hub Integration (.pyfun)

```ini
[LLM_PROVIDER.smollm]
active        = "true"
base_url      = "https://codey-lab-smollm-service.hf.space/v1"
env_key       = "TEST_TOKEN"
default_model = "smollm2-360m"
fallback_to   = "gemini"
[LLM_PROVIDER.smollm_END]
```

## Secrets (Space Settings)

| Secret | Required | Description |
|--------|----------|-------------|
| `HF_TOKEN` or `TEST_TOKEN` | optional | HF auth for dataset logging |
| `MODEL_REPO` | optional | Override base model (default: SmolLM2-360M) |
| `DATASET_REPO` | optional | HF dataset for logging (default: codey-lab/data.universal-mcp-hub) |
| `PRIVATE_MODEL_REPO` | optional | Private model repo for finetuning (default: codey-lab/model.universal-mcp-hub) |

## ADI Routing

| Decision | Action |
|----------|--------|
| `HIGH_PRIORITY` | SmolLM2 handles it |
| `MEDIUM_PRIORITY` | SmolLM2 handles it |
| `REJECT` | Returns improvement suggestions, logs to dataset |
| SmolLM2 fails | Returns 503 → Hub fallback chain kicks in |

## Training Utilities

```bash
python train.py --mode export    # export dataset to JSONL
python train.py --mode validate  # validate ADI weights
python train.py --mode finetune  # finetune (coming soon)
```

## Architecture

Part of [LLM-API-Gateway](https://github.com/VolkanSah/Multi-LLM-API-Gateway) ecosystem.
ADI idea: [Anti-Dump-Index](https://github.com/VolkanSah/Anti-Dump-Index)
