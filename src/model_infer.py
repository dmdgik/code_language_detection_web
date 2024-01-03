from typing import Any, Dict
from random import randint
import numpy as np
import torch
import time


def string_processing(
    string: str,
    tokenizer: Any,
    max_length: int,
    unk_tokens_ratio: float,
    use_part_text: bool,
    cls_token_id: int,
    pad_token_id: int,
    unk_token_id: int,
    next_line_token: str,
) -> tuple:
    """Converting string into input ids and attention mask with defined
    tokenizer and conditions.

    Args:
        string (str): string for processing
        tokenizer (Any): tokenizer
        max_length (int): maximum count of tokens
        unk_tokens_ratio (float): token dropout power
        use_part_text (bool): using only random snippet from provided string
        next_line_token (str): next line token from tokenizer (used only if use_part_text=True)
        cls_token_id (int): special classification token id from tokenizer
        pad_token_id (int): padding token id from tokenizer
        unk_token_id (int): unknown token id from tokenizer

    Returns:
        tuple: input_ids, attention_mask of tokenized string
    """
    string_tokens = " ".join(tokenizer.tokenize(string))

    if use_part_text:
        str_lines = string_tokens.split(next_line_token)
        count_str_lines = len(str_lines)
        start_line_idx = randint(0, count_str_lines - 1)
        end_line_idx = randint(start_line_idx + 1, count_str_lines)
        str_lines_selected = str_lines[start_line_idx:end_line_idx]
        string_tokens = next_line_token.join(str_lines_selected).strip()

    token_ids = tokenizer.convert_tokens_to_ids(string_tokens.split(" "))
    token_ids = [cls_token_id] + token_ids
    token_ids = token_ids[:max_length]
    input_ids = np.array(token_ids + (max_length - len(token_ids)) * [pad_token_id])

    input_attention_mask = (input_ids != pad_token_id) * 1

    if unk_tokens_ratio != 0.0:
        dropout_mask = np.random.choice(
            2, np.shape(input_ids), p=[1 - unk_tokens_ratio, unk_tokens_ratio]
        )
        dropout_mask &= input_ids != pad_token_id
        dropout_mask &= input_ids != cls_token_id

        input_ids = np.choose(
            dropout_mask, [input_ids, np.full_like(input_ids, unk_token_id)]
        )

    return input_ids, input_attention_mask


def model_infer(
    processed_str: str, model: Any, tokenizer: Any, languages_list_map: dict
) -> tuple:
    """Model infer ffunction

    Args:
        processed_str (str): string for predicting
        model (Any): model for predicting
        tokenizer (Any): tokenizer
        languages_list_map (dict): dict for convert predicted class num to string value

    Returns:
        tuple: language class num, language string, execution time
    """
    cls_token_id = tokenizer.cls_token_id
    pad_token_id = tokenizer.pad_token_id
    unk_token_id = tokenizer.unk_token_id
    next_line_token = tokenizer.tokenize("\n")[0]

    start_time = time.time()
    input_ids, input_attention_mask = string_processing(
        processed_str,
        tokenizer,
        512,
        0.0,
        False,
        cls_token_id,
        pad_token_id,
        unk_token_id,
        next_line_token,
    )
    input_ids = torch.tensor(input_ids, dtype=torch.int32)[None, :]
    input_attention_mask = torch.tensor(input_attention_mask, dtype=torch.int32)[
        None, :
    ]
    model_input = {"input_ids": input_ids, "attention_mask": input_attention_mask}
    with torch.no_grad():
        output = model(model_input)
    prediction_class = torch.argmax(output, dim=1).item()
    execution_time = time.time() - start_time

    language_str = [
        key for key, value in languages_list_map.items() if value == prediction_class
    ]
    language_str = language_str[0]

    return prediction_class, language_str, execution_time
