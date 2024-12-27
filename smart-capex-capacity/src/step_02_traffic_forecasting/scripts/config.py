# import packages
from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    """
    Loads a configuration file from the specified path and returns the
    configuration as a dictionary.

    Args:
    path: The path to the configuration file.

    Returns:
    dict: A dictionary containing the configuration data.
    """
    with open(path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    return configs

# configuartion file path
config_path = Path('config.yaml').resolve()

# load pipeline configuration
pipeline_config = load_config(config_path)
