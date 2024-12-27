# import packages
from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    """
    Load a configuration file from the specified path and return the configuration as a dictionary.

    Args:
    path (str): The path to the configuration file.

    Returns:
    dict: A dictionary containing the configuration data.
    """

    with open(path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

# configuration file path
config_path = Path('config.yaml').resolve()

# load pipeline configuration
pipeline_config = load_config(config_path)
