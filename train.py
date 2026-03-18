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
#   python train.py --mode finetune → finetune SmolLM2 on collected data (future)
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

import model as model_module
from adi import DumpindexAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train")


# =============================================================================
# Mode 1 — Export dataset to training format
# =============================================================================

def export_dataset(output_path: str = None):
    output = Path(output_path) if output_path else TRAIN_DATA
    """
    Export HF dataset logs to JSONL format for training.
    Filters: only HIGH_PRIORITY and MEDIUM_PRIORITY entries with actual responses.
    """
    logger.info("Loading dataset from HF...")
    entries = model_module.load_logs()

    if not entries:
        logger.warning("Dataset empty — nothing to export")
        return

    output = Path(output_path)
    count = 0

    with open(output, "w") as f:
        for entry in entries:
            # Only export entries where SmolLM2 actually responded
            if entry.get("adi_decision") == "REJECT":
                continue
            if not entry.get("response"):
                continue

            # Format as instruction tuning pair
            record = {
                "instruction": entry.get("system_prompt", "You are a helpful assistant."),
                "input":       entry.get("prompt", ""),
                "output":      entry.get("response", ""),
                "adi_score":   entry.get("adi_score"),
                "adi_decision": entry.get("adi_decision"),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info(f"Exported {count}/{len(entries)} entries → {output}")


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

    # Save results
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "accuracy": accuracy,
        "samples": len(labeled),
        "weights": analyzer.weights,
    }
    VALID_RESULT.write_text(json.dumps(result, indent=2))
    logger.info("Results saved → validation_results.json")


# =============================================================================
# Mode 3 — Finetune placeholder
# =============================================================================

def finetune():
    if not TRAIN_DATA.exists():
        logger.error(f"train_data.jsonl not found at {TRAIN_DATA}")
        return
    lines = TRAIN_DATA.read_text().strip().splitlines()
    logger.info(f"Training samples available: {len(lines)}")

    if len(lines) < 100:
        logger.warning(f"Only {len(lines)} samples — recommend 500+ for meaningful finetuning")

    # TODO: implement finetuning with transformers Trainer
    # Rough plan:
    #   1. Load base model via model.get_model_id()
    #   2. Tokenize train_data.jsonl
    #   3. TrainingArguments + Trainer
    #   4. Save to PRIVATE_MODEL repo via model.push_model_card()
    logger.info("Finetune placeholder — not yet implemented")
    logger.info("Next step: implement with transformers.Trainer or TRL SFTTrainer")


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
    parser.add_argument("--output", default="train_data.jsonl", help="Output file for export mode")
    args = parser.parse_args()

    if args.mode == "export":
        export_dataset(args.output)
    elif args.mode == "validate":
        validate_adi()
    elif args.mode == "finetune":
        finetune()
