# import packages
from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    """
    Load a configuration file from the specified path and returns the
    configuration as a dictionary.

    Args:
    path: The path to the configuration file.

    Returns:
    dict: A dictionary containing the configuration data.
    """
    with open(path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

# configuartion file path
prediction_config_path = Path('config/step_05_get_all_traffic_improvement_pipeline_for_prediction_config.yaml').resolve()

# load pipeline configuration
pipeline_config = load_config(prediction_config_path)
