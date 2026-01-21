import json
import json
import copy
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.models.api_model import api_client
from src.utils.text_process import get_think
from src.utils.prompt import LOGIC_REWARD_PROMPT

cur_api_client = api_client("local")
model_name = "openai/gpt-oss-20b"

jsonl_path = "results/SQA_qcm_a/qwen2.5_grpo_en_rain.jsonl"
cur_datas = []
with open(jsonl_path, 'r') as f:
    for line in f:
        cur_datas.append(json.loads(line))

def process_single_data(data):
    cur_data = copy.deepcopy(data)
    question_and_options = cur_data['prompt'].split('<|vision_end|>')[1].split('You FIRST think about the reasoning process')[0].strip().replace("\nAnswer with the option's letter from the given choices.", "")
    reasoning_process = get_think(cur_data['output'])
    query = LOGIC_REWARD_PROMPT.format(question_and_options=question_and_options, reasoning_process=reasoning_process)
    response = cur_api_client.call_text_api(query=query, 
                                            temperature=0,
                                            model=model_name)
    cur_data['fa_rp'] = response
    return cur_data

# Tune thread count based on API capacity (try 10-20 first)
max_workers = 100
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Use map to preserve ordering; deepcopy ensures thread safety
    results = list(tqdm(
        executor.map(process_single_data, cur_datas),
        total=len(cur_datas),
        desc="processing data concurrently"
    ))

# Save filtered data to save_path (same folder with /think_ans appended)
save_path = os.path.join(os.path.dirname(jsonl_path), 'think_ans', os.path.basename(jsonl_path).replace('.jsonl', f'-with_{model_name}_think_ans.jsonl'))
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    for data in results:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
