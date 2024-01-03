from typing import Any
from loguru import logger
from transformers import AutoTokenizer
import torch
import boto3
import os


def load_tokenizer(tokenizer_path: str) -> Any:
    """Tokenizer loading function

    Args:
        tokenizer_name (str): pretrained tokenizer name or path

    Raises:
        e: Exception

    Returns:
        Any: tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer {tokenizer_path} loaded successfully")
    except Exception as e:
        logger.error(f"Tokenizer {tokenizer_path} loading failed. Exception: {e}")
        raise e
    return tokenizer


def save_tokenizer(tokenizer: Any, path_to_save: str) -> None:
    """Tokenizer saving function

    Args:
        tokenizer (Any): tokenizer
        path_to_save (str): tokenizer saving path

    Raises:
        e: Exception
    """
    try:
        tokenizer.save_pretrained(path_to_save)
        logger.info(f"Tokenizer saved to {path_to_save} successfully")
    except Exception as e:
        logger.error(f"Tokenizer saving to {path_to_save} failed. Exception: {e}")
        raise e


def load_and_save_tokenizer(tokenizer_path: str, path_to_save: str) -> None:
    """Function for loading tokenizer and saving it locally

    Args:
        tokenizer_name (str): pretrained tokenizer name or path
        path_to_save (str): tokenizer saving path
    """
    tokenizer = load_tokenizer(tokenizer_path)
    save_tokenizer(tokenizer, path_to_save)


def load_model_torchscript(file_path: str, device="cpu") -> Any:
    """Function for loading torchscripted model

    Args:
        file_path (str): torchscripted model file path
        device (str, optional): model device placement. Defaults to "cpu".

    Returns:
        Any: torchscripted model
    """
    model = torch.jit.load(file_path)
    model.eval()
    model.to(torch.device(device))
    return model


def s3_download_model(
    s3_bucket_name: str,
    s3_model_file: str,
    local_model_file: str,
) -> None:
    """Function s3 model downloading

    Args:
        s3_bucket_name (str): AWS S3 bucket name
        s3_model_file (str): AWS S3 model file path
        local_model_file (str): local folder for saving model
    """
    print("loading model ...")
    if not os.path.isfile(local_model_file):
        logger.info("Model s3 downloading ...")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_S3_SECRET_ACCESS_KEY"],
        )
        s3.download_file(s3_bucket_name, s3_model_file, local_model_file)
        logger.info("Model s3 downloaded")
    else:
        logger.info("Model already exist")
