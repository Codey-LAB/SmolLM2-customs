# =============================================================================
# train.py
# Dataset Preparation + Finetuning Entry Point
# SmolLM2 Service Space
# Copyright 2026 - Volkan Kücükbudak
# Apache License V2 + ESOL 1.1
# =============================================================================
# Usage:
#   python train.py --mode export   → export HF dataset to training format
#   python train.py --mode validate → validate ADI weights against dataset
#   python train.py --mode finetune → finetune SmolLM2 on exported data
# =============================================================================
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# ── Path Resolution ───────────────────────────────────────────────────────────
# HF Spaces: /tmp/ (read-only filesystem)
# Local dev: current directory
_TMP = Path("/tmp") if os.getenv("SPACE_ID") else Path(".")

TRAIN_DATA   = _TMP / "train_data.jsonl"
VALID_RESULT = _TMP / "validation_results.json"
MODEL_OUTPUT = _TMP / "finetuned_model"

import model as model_module
from adi import DumpindexAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train")


# =============================================================================
# Mode 1 — Export dataset to training format
# =============================================================================

def export_dataset(output_path: str = None):
    """
    Export HF dataset logs to JSONL format for training.
    Includes HIGH_PRIORITY, MEDIUM_PRIORITY and BLOCKED entries.
    BLOCKED entries teach the model what to reject.
    REJECT entries (ADI noise/quality fail) are skipped — no response logged.
    """
    output = Path(output_path) if output_path else TRAIN_DATA

    logger.info("Loading dataset from HF...")
    entries = model_module.load_logs()

    # ── DEBUG: remove after fix ───────────────────────────────────────────
    #if entries:
        #logger.info(f"Keys: {list(entries[0].keys())}")
        #logger.info(f"Sample: {entries[0]}")
    # ─────────────────────────────────────────────────────────────────────

    if not entries:
        logger.warning("Dataset empty — nothing to export")
        return

    count = 0
    skipped = 0
    with open(output, "w") as f:
        for entry in entries:
            # Skip ADI-rejected entries — no meaningful response logged
            if entry.get("adi_decision") == "REJECT":
                skipped += 1
                continue
            if not entry.get("response"):
                skipped += 1
                continue

            # Format as instruction tuning pair
            # BLOCKED entries are included — model learns what to refuse
            record = {
                "instruction":  entry.get("system_prompt", "You are a helpful assistant."),
                "input":        entry.get("prompt", ""),
                "output":       entry.get("response", ""),
                "adi_score":    entry.get("adi_score"),
                "adi_decision": entry.get("adi_decision"),
                "is_safe":      entry.get("adi_decision") != "BLOCKED",
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info(f"Exported {count}/{len(entries)} entries → {output} (skipped: {skipped})")


# =============================================================================
# Mode 2 — Validate ADI weights against collected data
# =============================================================================

def validate_adi():
    """
    Run ADI weight validation against dataset.
    Uses entries that have human_label field (manually labeled).
    """
    logger.info("Loading dataset for ADI validation...")
    entries = model_module.load_logs()

    labeled = [(e["prompt"], e["human_label"]) for e in entries if e.get("human_label")]

    if not labeled:
        logger.warning("No labeled entries found — add 'human_label' field to dataset entries")
        logger.info("Expected labels: REJECT | MEDIUM_PRIORITY | HIGH_PRIORITY")
        return

    analyzer = DumpindexAnalyzer()
    accuracy = analyzer.validate_weights(labeled)
    logger.info(f"ADI Validation accuracy: {accuracy:.1%} on {len(labeled)} samples")

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "accuracy":  accuracy,
        "samples":   len(labeled),
        "weights":   analyzer.weights,
    }
    VALID_RESULT.write_text(json.dumps(result, indent=2))
    logger.info(f"Results saved → {VALID_RESULT}")


# =============================================================================
# Mode 3 — Finetune SmolLM2 with TRL SFTTrainer
# =============================================================================

def finetune():
    """
    Finetune SmolLM2 on exported dataset using TRL SFTTrainer.
    Requires export first + enough data (500+ samples recommended).
    On completion: pushes finetuned weights to private HF model repo.
    """
    if not TRAIN_DATA.exists():
        logger.error(f"train_data.jsonl not found at {TRAIN_DATA} — run export first")
        return

    lines = TRAIN_DATA.read_text().strip().splitlines()
    logger.info(f"Training samples available: {len(lines)}")

    if len(lines) < 10:
        logger.error(f"Too few samples ({len(lines)}) — aborting finetune")
        return

    if len(lines) < 500:
        logger.warning(f"Only {len(lines)} samples — recommend 500+ for meaningful finetuning")

    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        import torch
    except ImportError as e:
        logger.error(f"Missing dependency: {e} — run: pip install trl transformers datasets torch")
        return

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading training data...")
    records = [json.loads(l) for l in lines]

    def format_record(record):
        """Format record into chat template string."""
        instruction = record.get("instruction", "You are a helpful assistant.")
        user_input  = record.get("input", "")
        output      = record.get("output", "")
        return {
            "text": f"<|system|>\n{instruction}\n<|user|>\n{user_input}\n<|assistant|>\n{output}"
        }

    formatted = [format_record(r) for r in records]
    dataset   = Dataset.from_list(formatted)
    logger.info(f"Dataset ready: {len(dataset)} samples")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    model_id = model_module.get_model_id()
    kwargs   = model_module.get_model_kwargs()
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base model: {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model     = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Training config ───────────────────────────────────────────────────────
    # Conservative settings for CPU / low RAM (2-8GB)
    sft_config = SFTConfig(
        output_dir=str(MODEL_OUTPUT),
        num_train_epochs=3,
        per_device_train_batch_size=1,      # CPU friendly
        gradient_accumulation_steps=4,      # effective batch size = 4
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=False,                         # no GPU, no fp16
        bf16=False,
        dataloader_num_workers=0,           # HF Spaces: no multiprocessing
        report_to="none",                   # no wandb/tensorboard
        max_seq_length=512,                 # SmolLM2 context limit
        dataset_text_field="text",
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        #tokenizer=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting finetuning...")
    start = datetime.utcnow()
    trainer.train()
    duration = (datetime.utcnow() - start).total_seconds()
    logger.info(f"Training complete in {duration:.0f}s")

    # ── Save locally ──────────────────────────────────────────────────────────
    trainer.save_model(str(MODEL_OUTPUT))
    tokenizer.save_pretrained(str(MODEL_OUTPUT))
    logger.info(f"Model saved → {MODEL_OUTPUT}")

    # ── Push to HF private repo ───────────────────────────────────────────────
    token        = model_module.TOKEN
    private_repo = model_module.PRIVATE_MODEL

    if token and private_repo:
        logger.info(f"Pushing to HF: {private_repo}...")
        try:
            model.push_to_hub(private_repo, token=token, private=True)
            tokenizer.push_to_hub(private_repo, token=token, private=True)
            model_module.push_model_card({
                "model_id":       model_id,
                "samples":        len(dataset),
                "epochs":         3,
                "duration_sec":   int(duration),
                "finetuned_from": model_id,
            })
            logger.info(f"Model pushed → {private_repo}")
        except Exception as e:
            logger.error(f"Push failed: {type(e).__name__}: {e}")
    else:
        logger.warning("No token or private repo configured — skipping HF push")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolLM2 Training Utilities")
    parser.add_argument(
        "--mode",
        choices=["export", "validate", "finetune"],
        required=True,
        help="export: dump dataset to JSONL | validate: test ADI weights | finetune: train model"
    )
    parser.add_argument("--output", default=None, help="Output file for export mode (default: auto)")
    args = parser.parse_args()

    if args.mode == "export":
        export_dataset(args.output)
    elif args.mode == "validate":
        validate_adi()
    elif args.mode == "finetune":
        finetune()
