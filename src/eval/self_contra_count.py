import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.text_process import get_option_letter, get_answer


FILENAME_RE = re.compile(r"qwen2\.5_(?P<method>.+?)_en_(?P<data>.+?)-with", re.IGNORECASE)


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_filename(path: Path) -> Tuple[str, str]:
    match = FILENAME_RE.search(path.name)
    if not match:
        return "", path.stem
    return match.group("method").lower(), match.group("data")


def count_types(entries: List[Dict]) -> Dict[str, int]:
    counts = {"type_all": 0, "type_1": 0, "type_2": 0, "type_3": 0}
    for entry in entries:
        fa = get_option_letter(get_answer(entry.get("output", "")))
        fa_rp = (entry.get("fa_rp") or "").upper().strip()
        answer = (entry.get("answer") or "").upper().strip()

        if fa != fa_rp:
            counts["type_all"] += 1
            if fa_rp == "CANNOT BE DETERMINED":
                counts["type_3"] += 1
            else:
                if fa != answer and fa_rp == answer:
                    counts["type_1"] += 1
                elif fa == answer and fa_rp != answer:
                    counts["type_2"] += 1
                else:
                    counts["type_3"] += 1
    return counts


def summarize_file(path: Path) -> Dict[str, object]:
    entries = load_jsonl(path)
    total = len(entries)
    counts = count_types(entries)

    method, data_name = parse_filename(path)

    def proportion(value: int) -> float:
        return round(value / total, 6) if total else 0.0

    return {
        "method": method,
        "data": data_name,
        "type_all_count": counts["type_all"],
        "type_1_count": counts["type_1"],
        "type_2_count": counts["type_2"],
        "type_3_count": counts["type_3"],
        "type_all_proportion": proportion(counts["type_all"]),
        "type_1_proportion": proportion(counts["type_1"]),
        "type_2_proportion": proportion(counts["type_2"]),
        "type_3_proportion": proportion(counts["type_3"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count contradiction types for all jsonl files in a directory."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("results/SQA_qcm_a/think_ans"),
        help="Directory containing jsonl evaluation files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/SQA_qcm_a/think_ans_summary.csv"),
        help="Path to save the aggregated CSV.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found under {input_dir}")

    rows = [summarize_file(path) for path in jsonl_files]

    fieldnames = [
        "method",
        "data",
        "type_all_count",
        "type_1_count",
        "type_2_count",
        "type_3_count",
        "type_all_proportion",
        "type_1_proportion",
        "type_2_proportion",
        "type_3_proportion",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
