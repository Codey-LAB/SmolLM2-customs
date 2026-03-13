# =============================================================================
# model.py
# HuggingFace Model + Dataset Access Layer
# SmolLM2 Service Space
# Copyright 2026 - Volkan Kücükbudak
# Apache License V2 + ESOL 1.1
# =============================================================================
# Handles:
#   - Model loading (SmolLM2 from HF or private repo)
#   - Dataset read/write (private HF dataset)
#   - Token resolution (HF_TOKEN → TEST_TOKEN → None)
# =============================================================================

import os
import logging
from datetime import datetime
from typing import Optional
from huggingface_hub import HfApi, login
from datasets import load_dataset, Dataset

logger = logging.getLogger("model")

# ── Token Resolution ──────────────────────────────────────────────────────────
TOKEN = (
    os.environ.get("SMOLLM_API_KEY") or
    os.environ.get("HF_TOKEN") or
    os.environ.get("TEST_TOKEN") or
    os.environ.get("HUGGINGFACE_TOKEN") or
    os.environ.get("HF_API_TOKEN") or
    None
)

# ── Config from ENV ───────────────────────────────────────────────────────────
MODEL_REPO    = os.environ.get("MODEL_REPO", "HuggingFaceTB/SmolLM2-360M-Instruct")
DATASET_REPO  = os.environ.get("DATASET_REPO", "codey-lab/data.universal-mcp-hub")
PRIVATE_MODEL = os.environ.get("PRIVATE_MODEL_REPO", "codey-lab/model.universal-mcp-hub")

# ── HF API ────────────────────────────────────────────────────────────────────
_api: Optional[HfApi] = None

def get_api() -> Optional[HfApi]:
    """Returns authenticated HfApi instance or None if no token."""
    global _api
    if _api is None and TOKEN:
        try:
            login(token=TOKEN, add_to_git_credential=False)
            _api = HfApi(token=TOKEN)
            logger.info("HF API authenticated")
        except Exception as e:
            logger.warning(f"HF API auth failed: {type(e).__name__} — running unauthenticated")
    return _api


# =============================================================================
# Model Access
# =============================================================================

def get_model_id() -> str:
    """
    Returns model ID to load.
    Prefers private fine-tuned model only if it has actual weights (config.json with model_type).
    Falls back to base model if private repo is empty or not ready.
    """
    api = get_api()
    if api and PRIVATE_MODEL:
        try:
            files = api.list_repo_files(PRIVATE_MODEL, repo_type="model", token=TOKEN)
            has_config = "config.json" in list(files)
            if has_config:
                # Double-check it's a real model config, not just a README
                from huggingface_hub import hf_hub_download
                import json
                cfg_path = hf_hub_download(PRIVATE_MODEL, "config.json", token=TOKEN)
                cfg = json.loads(open(cfg_path).read())
                if "model_type" in cfg:
                    logger.info(f"Using private model: {PRIVATE_MODEL}")
                    return PRIVATE_MODEL
            logger.info(f"Private repo exists but has no weights yet — using base: {MODEL_REPO}")
        except Exception as e:
            logger.info(f"Private model check failed ({type(e).__name__}) — using base: {MODEL_REPO}")
    return MODEL_REPO


def get_model_kwargs() -> dict:
    """Returns kwargs for from_pretrained() calls."""
    kwargs = {}
    if TOKEN:
        kwargs["token"] = TOKEN
    return kwargs


# =============================================================================
# Dataset Access
# =============================================================================

def load_logs() -> list:
    """
    Load existing log entries from HF Dataset.
    Returns empty list if dataset doesn't exist yet.
    """
    if not TOKEN:
        logger.warning("No token — dataset read skipped")
        return []
    try:
        ds = load_dataset(DATASET_REPO, split="train", token=TOKEN)
        return ds.to_list()
    except Exception as e:
        logger.info(f"Dataset load: {type(e).__name__} — starting fresh")
        return []


def push_log(entry: dict) -> bool:
    """
    Append a log entry to HF Dataset and push.
    
    Args:
        entry: dict with prompt, adi, response, model, timestamp etc.
    
    Returns:
        True on success, False on failure.
    """
    if not TOKEN:
        logger.warning("No token — dataset push skipped")
        return False
    try:
        existing = load_logs()
        entry["timestamp"] = datetime.utcnow().isoformat()
        existing.append(entry)
        ds = Dataset.from_list(existing)
        ds.push_to_hub(DATASET_REPO, token=TOKEN, private=True)
        logger.info(f"Dataset updated — total entries: {len(existing)}")
        return True
    except Exception as e:
        logger.warning(f"Dataset push failed: {type(e).__name__}: {e}")
        return False


def push_model_card(info: dict) -> bool:
    """
    Update model card / metadata in private model repo.
    Useful for tracking which weights/config is deployed.
    """
    api = get_api()
    if not api:
        return False
    try:
        content = f"""---
language: en
license: apache-2.0
base_model: {MODEL_REPO}
---

# SmolLM2 Service

Base: `{MODEL_REPO}`  
Dataset: `{DATASET_REPO}`  
Last updated: {datetime.utcnow().isoformat()}

## Config
```json
{info}
```
"""
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="README.md",
            repo_id=PRIVATE_MODEL,
            repo_type="model",
            token=TOKEN,
        )
        logger.info(f"Model card updated: {PRIVATE_MODEL}")
        return True
    except Exception as e:
        logger.warning(f"Model card update failed: {type(e).__name__}: {e}")
        return False


# =============================================================================
# Health
# =============================================================================

def status() -> dict:
    """Returns model/dataset config status for health endpoint."""
    return {
        "token":         "set" if TOKEN else "missing",
        "model_repo":    MODEL_REPO,
        "private_model": PRIVATE_MODEL,
        "dataset_repo":  DATASET_REPO,
        "hf_api":        "authenticated" if get_api() else "unauthenticated",
    }
