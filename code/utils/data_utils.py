"""
data_utils.py
Utility functions for loading datasets and preparing text samples for training or inference.
Uses shared paths from paths.py for dataset caching and supports optional cache_dir from config.
"""

import os
from datasets import load_dataset, load_from_disk
from paths import DATASETS_DIR


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def get_local_dataset_path(dataset_name: str, cache_dir: str = None) -> str:
    """
    Build a safe local path for storing datasets based on their Hugging Face name.

    Args:
        dataset_name (str): Hugging Face dataset identifier (e.g., 'knkarthick/samsum').
        cache_dir (str | None): Optional cache directory override (e.g., from config).

    Returns:
        str: Absolute path to local dataset folder.
    """
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    base_dir = cache_dir or DATASETS_DIR
    return os.path.join(base_dir, safe_name)


def select_subset(dataset, n_samples, seed=42):
    """Select a subset of the dataset.
    If n_samples is "all" or None, return the entire dataset.
    Otherwise, sample n_samples examples.
    """
    if n_samples == "all" or n_samples is None:
        return dataset

    if n_samples > len(dataset):
        print(
            f"Requested {n_samples} samples but only {len(dataset)} available. Using all samples."
        )
        return dataset

    return dataset.shuffle(seed=seed).select(range(n_samples))
    """
    Select a subset of the dataset.
    If n_samples is "all" or None, return the entire dataset.
    Otherwise, sample n_samples examples.
    """
    if n_samples == "all" or n_samples is None:
        return dataset

    if n_samples > len(dataset):
        print(
            f"Requested {n_samples} samples but only {len(dataset)} available. Using all samples."
        )
        return dataset

    return dataset.shuffle(seed=seed).select(range(n_samples))



def get_field_names(cfg_dataset, default_input='dialogue', default_output='summary'):
    """Return input and output column names based on field_map.
    field_map maps generic names ('input', 'output') to dataset-specific column names.
    """
    field_map = cfg_dataset.get('field_map', {})
    input_col = field_map.get('input', default_input)
    output_col = field_map.get('output', default_output)
    return input_col, output_col
def load_and_prepare_dataset(cfg):
    """
    Load dataset splits according to configuration.
    Ensures the FULL dataset is cached, and subsets are selected per run.
    Supports both new-style ("dataset": {"splits": {...}}) and old-style (top-level keys) configs.
    """
    # -----------------------------------------------------------------------
    # Extract dataset configuration
    # -----------------------------------------------------------------------
    if "dataset" in cfg:
        cfg_dataset = cfg["dataset"]
        dataset_name = cfg_dataset["name"]
        splits_cfg = cfg_dataset.get("splits", {})
        n_train = splits_cfg.get("train", "all")
        n_val = splits_cfg.get("validation", "all")
        n_test = splits_cfg.get("test", "all")
        seed = cfg_dataset.get("seed", 42)
    elif "datasets" in cfg and isinstance(cfg["datasets"], list):
        cfg_dataset = cfg["datasets"][0]
        dataset_name = cfg_dataset["path"]
        n_train = cfg.get("train_samples", "all")
        n_val = cfg.get("val_samples", "all")
        n_test = cfg.get("test_samples", "all")
        seed = cfg.get("seed", 42)
        # Determine column names based on field_map
        input_col, output_col = get_field_names(cfg_dataset)
        dataset_type = cfg_dataset.get('type')
    else:
        raise KeyError(
            "Dataset configuration not found. Expected 'dataset' or 'datasets' key."
        )

    # -----------------------------------------------------------------------
    # Load or download full dataset
    # -----------------------------------------------------------------------
    os.makedirs(DATASETS_DIR, exist_ok=True)
    local_path = os.path.join(DATASETS_DIR, dataset_name.replace("/", "_"))

    if os.path.exists(local_path):
        print(f"Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
    else:
        print(f"⬇️  Downloading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(local_path)
        print(f"Full dataset saved locally to: {local_path}")
        # If the dataset is of type 'completion', create a unified 'text' field
        if dataset_type == 'completion':
            task_instruction = cfg.get('task_instruction', '')
            def make_text(example):
                return f"{task_instruction}\n\n## Dialogue:\n{example[input_col]}\n\n## Summary:\n{example[output_col]}"
            dataset = dataset.map(lambda ex: {'text': make_text(ex)}, remove_columns=[])


    # -----------------------------------------------------------------------
    # Handle variations in split keys and select subsets dynamically
    # -----------------------------------------------------------------------
    val_key = "validation" if "validation" in dataset else "val"

    train = select_subset(dataset["train"], n_train, seed=seed)
    val = select_subset(dataset[val_key], n_val, seed=seed)
    test = select_subset(dataset["test"], n_test, seed=seed)

    print(
        f"Loaded {len(train)} train / {len(val)} val / {len(test)} test samples (from full cache)."
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Prompt / Message Construction
# ---------------------------------------------------------------------------


def build_user_prompt(dialogue: str, task_instruction: str) -> str:
    """Construct a summarization-style prompt given a dialogue and instruction."""
    return f"{task_instruction}\n\n## Dialogue:\n{dialogue}\n## Summary:"


def build_messages_for_sample(sample, task_instruction, include_assistant=False):
    """
    Build a chat-style message list for a given sample, compatible with
    models that use chat templates (like Llama 3).
    """
    # If the dataset provides a pre‑built 'text' field (completion style), use it directly.
    if "text" in sample:
        messages = [{"role": "user", "content": sample["text"]}]
    else:
        messages = [{
            "role": "user",
            "content": build_user_prompt(sample["dialogue"], task_instruction),
        }]
    if include_assistant:
        # For chat style datasets, include the reference summary.
        messages.append({"role": "assistant", "content": sample["summary"]})
    return messages
