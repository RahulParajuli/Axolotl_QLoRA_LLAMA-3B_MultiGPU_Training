"""
evaluate_qlora.py
Evaluate a fine-tuned LoRA model on the SAMSum dataset.
Reuses shared utilities for config, dataset loading, and inference.
"""

import os
import json
import argparse
import torch
from dotenv import load_dotenv
from peft import PeftModel

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.inference_utils import generate_predictions, compute_rouge

load_dotenv()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_peft_model(cfg, adapter_path: str):
    """Load base model, attach LoRA adapters, and evaluate."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Validate adapter path
    # ----------------------------
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")

    # Results will be saved in the parent directory of the adapter
    results_dir = os.path.dirname(adapter_path)

    # ----------------------------
    # Model & Tokenizer
    # ----------------------------
    print(f"\nLoading base model: {cfg['base_model']}...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=False, padding_side="left", device_map=device
    )

    print(f"Loading fine-tuned LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ----------------------------
    # Dataset
    # ----------------------------
    print("\nLoading dataset...")
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"Validation set size: {len(val_data)} samples")

    # ----------------------------
    # Inference
    # ----------------------------
    print("\nGenerating summaries...")
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        batch_size=cfg.get("eval_batch_size", 4),
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\nComputing ROUGE metrics...")
    scores = compute_rouge(preds, val_data)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"ROUGE-1: {scores['rouge1']:.4f} ({scores['rouge1']:.2%})")
    print(f"ROUGE-2: {scores['rouge2']:.4f} ({scores['rouge2']:.2%})")
    print(f"ROUGE-L: {scores['rougeL']:.4f} ({scores['rougeL']:.2%})")
    print("=" * 50)

    # ----------------------------
    # Save Outputs
    # ----------------------------
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "eval_results.json")
    preds_path = os.path.join(results_dir, "predictions.jsonl")

    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "base_model": cfg["base_model"],
        "adapter_path": adapter_path,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            json.dump(
                {
                    "dialogue": val_data[i]["dialogue"],
                    "reference": val_data[i]["summary"],
                    "prediction": pred,
                },
                f,
            )
            f.write("\n")

    print(f"\n✅ Saved results to {results_path}")
    print(f"✅ Saved predictions to {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA fine-tuned model on SAMSum"
    )
    parser.add_argument(
        "--adapter_path",
        "-a",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (e.g., data/outputs/accelerate_ddp/llama-3.2-8b-2-gpus/lora_adapters)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config()

    # Evaluate
    scores, preds = evaluate_peft_model(cfg, args.adapter_path)

    print("\n✅ Evaluation complete.")
    print("\nSample prediction:")
    print(f"{preds[0][:150]}...")


if __name__ == "__main__":
    main()
