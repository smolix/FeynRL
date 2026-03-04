import re
from typing import Any, Dict
import torch


def normalize_answer(answer_str):
    if answer_str is None:
        return None
    return str(answer_str).replace(",", "").replace("$", "").replace("\n", "").strip()


def extract_solution(solution_str, clip_chars=300):
    if solution_str is None:
        return None
    solution_str = str(solution_str)
    if len(solution_str) > clip_chars:
        solution_str = solution_str[-clip_chars:]

    match = re.search(r"####\s*(-?[0-9.,]+)", solution_str)
    if match is None:
        return None
    return normalize_answer(match.group(1))


def compute_score(prompt_data: Dict[str, Any], response_data: Dict[str, Any]):
    """
    Reward is 1.0 on exact final-answer match (using #### format), else 0.0.
    """
    token_ids = list(response_data.token_ids)
    is_per_token = False
    r = torch.zeros((len(token_ids),), dtype=torch.float32)

    print("response.text", response_data.text)
    

    if len(token_ids) == 0:
        return r, is_per_token

    predicted = extract_solution(response_data.text)
    ground_truth = normalize_answer(prompt_data.get("solution"))

    if predicted is not None and ground_truth is not None and predicted == ground_truth:
        r[-1] = 1.0
    else:
        r[-1] = 0.0

    return r, is_per_token
