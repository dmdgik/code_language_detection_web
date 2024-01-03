import os
from .files_io_utils import read_yaml


def set_env_secrets(secrets_config_path: str):
    """Set secrets variables

    Args:
        secrets_config_path (str): path with secrets config
    """
    secrets_config = read_yaml(secrets_config_path)
    
    for secret_name, secret_value in secrets_config.items():
        os.environ[secret_name] = secret_value

