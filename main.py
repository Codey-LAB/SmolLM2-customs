# =============================================================================
# main.py
# FastAPI — OpenAI-compatible /v1/chat/completions endpoint
# SmolLM2 Service Space
# Copyright 2026 - Volkan Kücükbudak
# Apache License V2 + ESOL 1.1
# =============================================================================

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
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
    """Validate Bearer token. Skipped if API_KEY secret not set (dev mode)."""
    if not _API_KEY:
        return
    if authorization != f"Bearer {_API_KEY}":
        logger.warning("Unauthorized request — invalid or missing token")
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SmolLM2 Service starting ===")
    logger.info(f"Model config: {model_module.status()}")
    smollm.load()
    yield
    logger.info("=== SmolLM2 Service stopped ===")

app = FastAPI(title="SmolLM2 Service", version="1.0.0", lifespan=lifespan)


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
    return {
        "service": "SmolLM2 Service",
        "model":   smollm.device_info(),
        "ready":   smollm.is_ready(),
        "auth":    "protected" if _API_KEY else "open",
        "docs":    "/docs",
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


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

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
        model_module.push_log({
            "prompt":        user_prompt,
            "system_prompt": system_prompt,
            "adi_score":     adi_result["adi"],
            "adi_decision":  decision,
            "adi_metrics":   adi_result["metrics"],
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
        raise HTTPException(
            status_code=503,
            detail={
                "error":        "smollm_unavailable",
                "adi_decision": decision,
                "message":      "Route to next provider in fallback chain",
            }
        )

    # ── Log to Dataset ────────────────────────────────────────────────────────
    model_module.push_log({
        "prompt":        user_prompt,
        "system_prompt": system_prompt,
        "adi_score":     adi_result["adi"],
        "adi_decision":  decision,
        "adi_metrics":   adi_result["metrics"],
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
