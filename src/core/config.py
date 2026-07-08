import json
import os

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)

    return config
