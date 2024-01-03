from src.loading_utils import s3_download_model, load_and_save_tokenizer
from src.set_secrets import set_env_secrets
from src.files_io_utils import read_yaml
from loguru import logger


if __name__ == "__main__":
    APP_CONFIG_PATH = "./configs/app.yaml"
    SECRETS_CONFIG_PATH = "./configs/secrets.yaml"

    set_env_secrets(SECRETS_CONFIG_PATH)
    app_config = read_yaml(APP_CONFIG_PATH)

    s3_download_model(
        app_config["s3_bucket_name"],
        app_config["s3_model_file"],
        app_config["local_model_file"],
    )

    logger.log("INFO", "MODEL DOWNLOADED")

    load_and_save_tokenizer(app_config["tokenizer"], app_config["tokenizer_path"])
    
    logger.log("INFO", "TOKENIZER DOWNLOADED")

