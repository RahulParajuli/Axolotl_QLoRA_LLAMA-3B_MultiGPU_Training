# Helper utilities for fine‑tuning an instruction‑tuned model on the SamSum dataset
# using the Axolotl QLoRA framework.
# All functions are self‑contained and do not modify existing utilities.

"""
Helper utilities for fine‑tuning an instruction‑tuned model on the SamSum dataset
using the Axolotl QLoRA framework.
All functions are self‑contained and do not modify existing utilities.
"""

from typing import Tuple
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_samsum_dataset(tokenizer, max_length: int = 1024) -> Dataset:
    """Load the SamSum dataset and format it for an instruction‑tuned model.

    Returns a Dataset with columns ``input_ids`` and ``labels`` ready for
    ``Trainer``.  The instruction prompt is fixed to:
        "Summarize the following dialogue:"
    """
    raw = load_dataset("samsum")
    instruction = "Summarize the following dialogue:"

    def format_example(example):
        # Build the user‑side prompt
        prompt = f"{instruction}\n\n{example['dialogue']}"
        # Tokenize prompt
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Tokenize target summary
        label_ids = tokenizer(
            example["summary"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )["input_ids"]
        tokenized["labels"] = label_ids
        return tokenized

    return raw.map(format_example, remove_columns=raw["train"].column_names)


def get_instruct_model(
    model_name: str = "meta-llama/Meta-Llama-3B-Instruct",
    quantization_bits: int = 4,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load a quantized instruction‑tuned model suitable for QLoRA.
    Returns the model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_cfg = None
    if quantization_bits == 4:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return model, tokenizer
