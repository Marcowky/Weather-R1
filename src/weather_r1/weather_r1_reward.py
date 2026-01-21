import re
import copy
import json
import os
from functools import partial
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from src.utils.prompt import LOGIC_REWARD_PROMPT
from src.utils.text_process import get_think, get_answer, get_option_letter


DEFAULT_REWARD_WEIGHTS = {
    "format": 0.1,
    "logic": 0.3,
    "accuracy": 0.6,
}
DEFAULT_CLIENT_MODEL = "openai/gpt-oss-20b"

DEFAULT_API_KEY = "EMPTY"
DEFAULT_API_BASE = "http://0.0.0.0:50907/v1"
client = OpenAI(
    api_key=os.getenv("WEATHER_R1_OPENAI_API_KEY", DEFAULT_API_KEY),
    base_url=os.getenv("WEATHER_R1_OPENAI_API_BASE", DEFAULT_API_BASE),
)


def _resolve_client_model(model: Optional[str]) -> str:
    """Choose which client model to call."""
    return model or os.getenv("WEATHER_R1_CLIENT_MODEL", DEFAULT_CLIENT_MODEL)


def _resolve_reward_weights(weights: Optional[Union[str, Dict[str, float]]]) -> Dict[str, float]:
    """Merge provided weights (CLI/env) with defaults and sanitize values."""
    merged = DEFAULT_REWARD_WEIGHTS.copy()
    source = weights

    if isinstance(source, str):
        try:
            source = json.loads(source)
        except Exception:
            source = None

    if isinstance(source, dict):
        for key, value in source.items():
            if key in merged:
                try:
                    merged[key] = float(value)
                except (TypeError, ValueError):
                    pass

    # Ensure reward weights sum to 1.0
    total_weight = sum(merged.values())
    if not abs(total_weight - 1.0) < 1e-6:
        raise ValueError(f"Reward weights must sum to 1.0, but got {total_weight}")

    return merged


def call_client_model(query: str, model: Optional[str]) -> str:
    chat_response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user", 
            "content": query
        }],
        temperature=0,
        max_tokens=512,
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return chat_response.choices[0].message.content


def format_reward(response: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    # Count <think> and <answer> occurrences
    single_think = response.count("<think>") == 1 and response.count("</think>") == 1
    single_answer = response.count("<answer>") == 1 and response.count("</answer>") == 1

    # Check presence of <think> and <answer> tags
    has_think = bool(re.search(r'^<think>.*?</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>$', response, re.DOTALL))

    # Check whether both <think> and <answer> tags exist in order
    has_think_answer = bool(re.search(r"^<think>.*?</think>\s*<answer>.*?</answer>$", response, re.DOTALL))
    
    # Compute score
    score = 0.0
    if has_think and single_think:
        score += 0.25
    if has_answer and single_answer:
        score += 0.25
    if has_think_answer and single_think and single_answer:
        score += 0.5
    
    return score


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Reward function that checks if the completion matches the correct answer choice (A/B/C/D)."""
    reward = 0.0
    
    try:
        content_answer = get_answer(response)
        # Extract a single option letter (A/B/C/D) via regex
        student_answer = get_option_letter(content_answer)

        # Normalize ground_truth to uppercase
        ground_truth = ground_truth.upper().strip()
        
        # Compare answers
        if student_answer == ground_truth:
            # If content_answer is already a single option letter, grant full reward
            if content_answer == student_answer:
                reward = 1.0
            else:
                reward = 0.5
                
    except Exception:
        pass
        
    return reward


def logic_reward(problem: str, response: str, model: Optional[str]) -> float:
    reward = 0.0
    think_answer = ""

    try:
        question_and_options = problem.replace("<image>", "")
        reasoning_process = get_think(response)
        query = LOGIC_REWARD_PROMPT.format(question_and_options=question_and_options, reasoning_process=reasoning_process)
        think_answer = call_client_model(query, model=model)

        if "Cannot be determined".lower() not in think_answer.lower():
            think_answer = get_option_letter(think_answer)
        student_answer = get_option_letter(get_answer(response))

        if think_answer == student_answer:
            reward = 1.0

    except Exception:
        pass
    
    return reward, think_answer


def _process_single_reward_input(reward_input: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    """Helper function to process a single reward input."""
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    
    format_score = format_reward(response)
    if format_score != 1.0:
        logic_score = 0.0
        think_answer = "Format error"
    else:
        logic_score, think_answer = logic_reward(reward_input["problem"], response, model)
    accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
    
    return {
        "reward_input": reward_input,
        "scores": {
            "format": format_score,
            "logic": logic_score,
            "accuracy": accuracy_score
        },
        "think_answer": think_answer
    }


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    cur_stat: str,
    cur_step: int,
    save_path: str,
    reward_weights: Optional[Union[str, Dict[str, float]]] = None,
    client_model_name: Optional[str] = None,
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    resolved_weights = _resolve_reward_weights(reward_weights)
    resolved_model = _resolve_client_model(client_model_name)

    # Process inputs in parallel with a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=256) as executor:
        process_fn = partial(_process_single_reward_input, model=resolved_model)
        results = list(executor.map(process_fn, reward_inputs))

    scores = []
    save_datas = []
    
    for result in results:
        cur_result = copy.deepcopy(result)
        del cur_result["reward_input"]["response_length"]

        # Compute weighted overall score
        overall_score = sum(cur_result['scores'][reward] * weight for reward, weight in resolved_weights.items())
        cur_result['scores']["overall"] = overall_score if cur_stat != 'val' else cur_result['scores']["accuracy"]

        # Persist detailed reward data
        save_datas.append(cur_result)
        # Persist scores only
        scores.append(cur_result["scores"])

    # Build save path for detailed scores
    base_save_path = os.path.join(save_path, 'detailed_scores', f"stat_{cur_stat}_step_{cur_step}")
    # Create directory if missing
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
    # If file exists, increment suffix to avoid overwrite
    counter = 0
    json_save_path = f"{base_save_path}_{counter}.json"
    while os.path.exists(json_save_path):
        counter += 1
        json_save_path = f"{base_save_path}_{counter}.json"
    # Save detailed results to json
    with open(json_save_path, "w") as f:
        json.dump(save_datas, f, indent=4, ensure_ascii=False)

    return scores
