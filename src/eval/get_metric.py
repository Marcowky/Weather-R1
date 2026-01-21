import json
import os
from collections import defaultdict

from src.utils.text_process import get_answer, get_option_letter

folders = "results/SQA_qcm_a"

csv_path = os.path.join(folders, "summary.csv")
# Open CSV for writing results
csv_file = open(csv_path, "w", encoding="utf-8")
rows = []

for file in os.listdir(folders):
    if file.endswith('.jsonl'):
        jsonl_res = []
        with open(os.path.join(folders, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                jsonl_res.append(line)

            correct_num = defaultdict(int)
            total_num = defaultdict(int)
            type_wrong = defaultdict(set)

            if "grpo" in file or "dapo" in file and "major" not in file:
                for data in jsonl_res:
                    category = data['category']
                    content_answer = get_option_letter(get_answer(data['output']))
                    if content_answer in ['A', 'B', 'C', 'D']:
                        total_num[category] += 1
                    if content_answer == data['answer']:
                        correct_num[category] += 1
                    else:
                        type_wrong[category].add(data['id'])
            else:
                for data in jsonl_res:
                    category = data['category']
                    if data['output'][0] in ['A', 'B', 'C', 'D']:
                        total_num[category] += 1
                    if data['output'][0] == data['answer']:
                        correct_num[category] += 1
                    else:
                        type_wrong[category].add(data['id'])

            row_parts = [file.replace('.jsonl', '')]

            for k, v in correct_num.items():
                # print(k, v, total_num[k], v/total_num[k])
                row_parts.append(f"{v}")
                row_parts.append(f"{total_num[k]}")
                row_parts.append(f"{v/total_num[k]}")

            # print("total", sum(correct_num.values()), sum(total_num.values()), sum(correct_num.values())/sum(total_num.values()))
            print(file)
            row_parts.append(f"{sum(correct_num.values())}")
            row_parts.append(f"{sum(total_num.values())}")
            row_parts.append(f"{sum(correct_num.values())/sum(total_num.values())}")
            rows.append(row_parts)

for row in sorted(rows, key=lambda x: x[0]):
    csv_file.write(",".join(row))
    csv_file.write("\n")
csv_file.close()
