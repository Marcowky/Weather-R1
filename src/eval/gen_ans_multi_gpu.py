import os
import json
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.eval.weatherqa_dataset_loader import WeatherR1Dataset
from src.models.load_model import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    # parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--prompt-type", type=str, default=None)
    parser.add_argument("--language", type=str, default='cn')
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    args = parser.parse_args()
    return args


def eval_model(args):

    weather_r1_dataset = WeatherR1Dataset(weather_r1_path=args.data_file, image_rft_path=args.image_folder, prompt_type=args.prompt_type, language=args.language)
    
    # Split data into contiguous chunks instead of striding
    total_samples = len(weather_r1_dataset.data_list)
    samples_per_gpu = total_samples // args.gpu_num
    start_idx = args.gpu_id * samples_per_gpu
    if args.gpu_id == args.gpu_num - 1:  # Last GPU handles any remainder
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_gpu
    weather_r1_dataset.data_list = weather_r1_dataset.data_list[start_idx:end_idx]

    weather_r1_loader = DataLoader(weather_r1_dataset)

    cur_model = load_model(args.model_name, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    answers_file = args.answers_file
    # If file exists, append model name suffix
    if os.path.exists(answers_file):
        answers_file = answers_file.replace('.jsonl', '') + "_" + args.model_name + ".jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for _, data in tqdm(enumerate(weather_r1_loader), total=len(weather_r1_loader)):
        qs = data["prompt"][0]
        image_path = data["image_path"][0]
        answer = data["answer"][0]
        category = data["category"][0]
        idx = data["id"].tolist()[0]

        output, prompt = cur_model.generate(question=qs, image_path=image_path, temperature=args.temperature)

        res = {"id": idx,
            "prompt": prompt,
            "output": output,
            "answer": answer,
            "category": category,
            "model": args.model_name}
        ans_file.write(json.dumps(res, ensure_ascii=False) + '\n')
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    eval_model(args)
