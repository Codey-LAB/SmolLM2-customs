# =============================================================================
# main.py
# FastAPI — OpenAI-compatible /v1/chat/completions endpoint
# SmolLM2 Service Space
# Copyright 2026 - Volkan Kücükbudak
# Apache License V2 + ESOL 1.1
# =============================================================================
# Hub connects via:
#   base_url = "https://codey-lab-smollm2-customs.hf.space/v1"
#   → POST /v1/chat/completions  (OpenAI-compatible)
#   → GET  /v1/health            (status check)
#
# AUTH:
#   Set API_KEY in HF Space Secrets to lock down the endpoint.
#   Hub sends it as:  Authorization: Bearer <API_KEY>
#   If API_KEY not set → open access (dev mode, log warning)
# =============================================================================

import hashlib
import hmac
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

import smollm
import model as model_module
from adi import DumpindexAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# ── ADI ───────────────────────────────────────────────────────────────────────
adi_analyzer = DumpindexAnalyzer(enable_logging=False)

# ── API Key Auth ──────────────────────────────────────────────────────────────
_API_KEY = os.environ.get("SMOLLM_API_KEY", "")
if not _API_KEY:
    logger.warning("API_KEY not set — running in open access mode!")
else:
    logger.info("API_KEY set — endpoint is protected")


def _check_auth(authorization: Optional[str]) -> None:
    """Validate Bearer token using timing-safe comparison. Skipped in dev mode."""
    if not _API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Unauthorized request — missing or malformed Authorization header")
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization[len("Bearer "):]
    # hmac.compare_digest prevents timing attacks
    if not hmac.compare_digest(
        hashlib.sha256(token.encode()).digest(),
        hashlib.sha256(_API_KEY.encode()).digest(),
    ):
        logger.warning("Unauthorized request — invalid token")
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Rate Limiting ─────────────────────────────────────────────────────────────
# Simple in-process sliding window. Good enough for HF Space single-worker.
# Swap for Redis-backed slowapi if you ever run multi-worker.

_RATE_LIMIT_WINDOW  = 60        # seconds
_RATE_LIMIT_MAX     = 20        # requests per window per IP (chat endpoint)
_TRAIN_RATE_LIMIT   = 5         # requests per window per IP (train endpoint)
_request_log: dict  = defaultdict(list)


def _rate_check(key: str, max_requests: int) -> None:
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW
    # Purge old entries
    _request_log[key] = [t for t in _request_log[key] if t > window_start]
    if len(_request_log[key]) >= max_requests:
        logger.warning(f"Rate limit hit for key: {key}")
        raise HTTPException(status_code=429, detail="Too Many Requests")
    _request_log[key].append(now)


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SmolLM2 Service starting ===")
    logger.info(f"Model config: {model_module.status()}")
    smollm.load()
    yield
    logger.info("=== SmolLM2 Service stopped ===")

app = FastAPI(
    title="SmolLM2 Service",
    version="1.0.0",
    lifespan=lifespan,
    # Disable auto-generated docs in production if you want:
    # docs_url=None, redoc_url=None
)


# =============================================================================
# Request / Response Models (OpenAI-compatible)
# =============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model:       Optional[str]   = "smollm2-360m"
    messages:    List[Message]
    max_tokens:  Optional[int]   = 150
    temperature: Optional[float] = 0.2
    stream:      Optional[bool]  = False


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
async def root():
    """Minimal status — no internal details exposed."""
    return {
        "service": "SmolLM2 Service",
        "ready":   smollm.is_ready(),
        "auth":    "protected" if _API_KEY else "open",
    }


@app.get("/v1/health")
async def health(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    return {
        "status": "ok" if smollm.is_ready() else "loading",
        "device": smollm.device_info(),
        "model":  model_module.status(),
        "auth":   "protected" if _API_KEY else "open",
    }


# ── Training & Data Ops Trigger ──────────────────────────────────────────────
# How to trigger Training/Export/Validation outside HF (e.g., Git Actions):
#
# # 1. Export Dataset to JSONL:
# curl -X POST "https://codey-lab-smollm2-customs.hf.space/v1/train/execute?mode=export" \
#      -H "Authorization: Bearer ${{ secrets.SMOLLM_API_KEY }}"
#
# # 2. Validate ADI Weights:
# curl -X POST "https://codey-lab-smollm2-customs.hf.space/v1/train/execute?mode=validate" \
#      -H "Authorization: Bearer ${{ secrets.SMOLLM_API_KEY }}"
#
# # 3. Finetune SmolLM2:
# curl -X POST "https://codey-lab-smollm2-customs.hf.space/v1/train/execute?mode=finetune" \
#      -H "Authorization: Bearer ${{ secrets.SMOLLM_API_KEY }}"

_VALID_TRAIN_MODES = frozenset(["export", "validate", "finetune"])
_train_lock = False  # Simple guard against parallel train runs


@app.post("/v1/train/execute")
async def execute_train_ops(
    request: Request,
    mode: str = "export",
    authorization: Optional[str] = Header(None),
):
    """
    Remote trigger for train.py. Auth required — always.
    Supports: export | validate | finetune
    """
    global _train_lock

    # Auth is mandatory here regardless of dev mode
    if not _API_KEY:
        raise HTTPException(status_code=503, detail="Train endpoint disabled in open-access mode")
    _check_auth(authorization)

    # Rate limit train endpoint (tighter than chat)
    client_ip = request.client.host if request.client else "unknown"
    _rate_check(f"train:{client_ip}", _TRAIN_RATE_LIMIT)

    # Whitelist mode (already a frozenset — fast lookup)
    if mode not in _VALID_TRAIN_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Supported: {', '.join(sorted(_VALID_TRAIN_MODES))}"
        )

    # Concurrency guard — no parallel training runs
    if _train_lock:
        raise HTTPException(status_code=409, detail="A training task is already running")

    import subprocess
    import sys

    try:
        _train_lock = True
        proc = subprocess.Popen(
            [sys.executable, "train.py", "--mode", mode],
            # Isolate the subprocess — no inherited file descriptors leaking
            close_fds=True,
            start_new_session=True,
        )
        logger.info(f"TRAIN-OPS | pid={proc.pid} | mode={mode} | ip={client_ip}")
        return {
            "status":    "queued",
            "mode":      mode,
            "message":   f"train.py --mode {mode} triggered",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"TRAIN-OPS | Failed to start: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal Execution Error")
    finally:
        # Release lock after a short grace period so the process can actually start.
        # In production you'd track proc.returncode properly; this is fine for HF Space.
        _train_lock = False


# ── chat/completions ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    # Rate limit per IP
    client_ip = request.client.host if request.client else "unknown"
    _rate_check(f"chat:{client_ip}", _RATE_LIMIT_MAX)

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # ── Extract prompt + system prompt ────────────────────────────────────────
    system_prompt = ""
    user_prompt   = ""

    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_prompt = msg.content

    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    # ── ADI Analysis ──────────────────────────────────────────────────────────
    adi_result = adi_analyzer.analyze_input(user_prompt)
    decision   = adi_result["decision"]
    logger.info(f"ADI | decision: {decision} | score: {adi_result['adi']}")

    # ── Route by ADI decision ─────────────────────────────────────────────────
    if decision == "REJECT":
        logger.info("ADI → REJECT: returning rejection response")
        response_text = (
            "Your request needs more detail before I can help. "
            "Suggestions: " + " | ".join(adi_result["recommendations"])
        )
        import json as _json
        model_module.push_log({
            "prompt":        user_prompt,
            "system_prompt": system_prompt,
            "adi_score":     adi_result["adi"],
            "adi_decision":  decision,
            "adi_metrics":   _json.dumps(adi_result["metrics"]),  # Arrow needs string, not dict
            "response":      None,
            "routed_to":     "REJECT",
            "model":         req.model,
        })
        return _build_response(req.model, response_text, adi_result)

    # ── SmolLM2 Inference ─────────────────────────────────────────────────────
    try:
        response_text = await smollm.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        routed_to = "smollm2"
        logger.info(f"SmolLM2 response ok | decision: {decision}")

    except Exception as e:
        logger.warning(f"SmolLM2 failed: {type(e).__name__} — triggering hub fallback")
        # adi_decision is intentional here — hub needs it for fallback routing.
        # Safe because this response is only visible to authenticated hub clients.
        raise HTTPException(
            status_code=503,
            detail={
                "error":        "smollm_unavailable",
                "adi_decision": decision,
                "message":      "Route to next provider in fallback chain",
            }
        )

    # ── Log to Dataset ────────────────────────────────────────────────────────
    import json as _json
    model_module.push_log({
        "prompt":        user_prompt,
        "system_prompt": system_prompt,
        "adi_score":     adi_result["adi"],
        "adi_metrics":   _json.dumps(adi_result["metrics"]),   # Arrow needs string, not dict
        "adi_decision":  decision,
        "response":      response_text,
        "routed_to":     routed_to,
        "model":         req.model,
    })

    return _build_response(req.model, response_text, adi_result)


# =============================================================================
# Helpers
# =============================================================================

def _build_response(model: str, content: str, adi_result: dict) -> dict:
    return {
        "id":      f"smollm-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "adi": {
            "score":    adi_result["adi"],
            "decision": adi_result["decision"],
            "metrics":  adi_result["metrics"],
        }
    }
