"""
Shared utilities for inference and evaluation — text generation and metric computation.
"""

import torch
from tqdm import tqdm
from transformers import pipeline
import evaluate


def generate_predictions(
    model,
    tokenizer,
    dataset,
    task_instruction,
    num_samples=None,
    batch_size=8,
    max_new_tokens=256,
):
    """
    Generate model predictions for a dataset (e.g., summaries).

    Args:
        model: The loaded model (base or fine-tuned).
        tokenizer: Corresponding tokenizer.
        dataset: Hugging Face dataset split containing 'dialogue' and 'summary'.
        task_instruction (str): Instruction prefix for generation.
        num_samples (int, optional): Number of samples to evaluate.
        batch_size (int): Number of examples per inference batch.
        max_new_tokens (int): Max tokens to generate per sample.

    Returns:
        list[str]: Generated summaries.
    """
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    # Prepare prompts
    prompts = []
    for sample in dataset:
        user_prompt = (
            f"{task_instruction}\n\n"
            f"## Dialogue:\n{sample['dialogue']}\n"
            "## Summary:"
        )
        messages = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype="auto",
        do_sample=False,
    )

    # Generate predictions
    preds = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
        batch = prompts[i : i + batch_size]
        outputs = pipe(batch, max_new_tokens=max_new_tokens, return_full_text=False)
        preds.extend([o[0]["generated_text"].strip() for o in outputs])

    return preds


def compute_rouge(predictions, samples):
    """
    Compute ROUGE scores between predictions and reference summaries.

    Args:
        predictions (list[str]): Model-generated outputs.
        samples (datasets.Dataset): Dataset containing reference 'summary' field.

    Returns:
        dict: ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    rouge = evaluate.load("rouge")
    references = [s["summary"] for s in samples]
    return rouge.compute(predictions=predictions, references=references)
