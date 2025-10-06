from typing import Dict, Optional, Union
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(config_file: Union[Path, str]) -> Optional[Dict]:
    """
    Load configuration from a local file.

    :param config_file: Path to the local file
    :return: Dictionary with the loaded configuration
    """
    try:
        config_data: Optional[Dict] = None
        logger.info(f"Loading config from local file system: {config_file}")
        content = Path(config_file).read_text()
        config_data = yaml.safe_load(content)
        logger.info(f"Loaded config from local file system: {config_data}")
    except Exception as e:
        logger.error(f"Error loading config from local file system: {e}")
        config_data = None
    return config_data