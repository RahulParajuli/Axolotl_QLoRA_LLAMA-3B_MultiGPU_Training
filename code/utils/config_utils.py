"""
config_utils.py
Shared utilities for loading and managing YAML configuration files.
"""

import yaml
from paths import CONFIG_FILE_PATH


def load_config(config_path: str = CONFIG_FILE_PATH):
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
