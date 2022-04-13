#!/usr/bin/python
import os
import json
import argparse
from unittest import result
from tabulate import tabulate

result_file_name = "all_results.json"
result_types = [
    "eval-clean-clean",
    "eval-backdoored-clean",
    "eval-clean-poisioned",
    "eval-backdoored-poisioned"
]
glue_task_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
downstreams = ["glue", "ner", "qa"]
# 脚本所在目录的绝对路径
script_path = os.path.split(os.path.realpath(__file__))[0]

def glue_results(task_name: str = None):
    base_dir = os.path.join(script_path, "glue/result")
    result_dirs = sorted(os.listdir(base_dir))
    results = []
    for result_dir in result_dirs:
        result_task_name = result_dir.split("-")[0]
        if task_name != None and result_task_name != task_name:
            continue
        result = [result_dir]
        result_path = os.path.join(base_dir, result_dir)
        for result_type in result_types:
            result_file = os.path.join(result_path, result_type, result_file_name)
            # 可能存在某项结果不存在
            if not os.path.exists(result_file):
                result.append("")
                continue
            result_json = json.load(open(result_file))
            # 结果为 Accuracy
            if result_task_name in ("sst2", "qnli", "rte", "wnli"):
                accuracy = result_json["eval_accuracy"] * 100
                result.append(f"{accuracy:.2f}")
            # 结果为 Matthews corr
            elif result_task_name == "cola":
                matthews_correlation = result_json["eval_matthews_correlation"] * 100
                result.append(f"{matthews_correlation:.2f}")
            # 结果为 F1/Accuracy
            elif result_task_name == "mrpc":
                f1 = result_json["eval_f1"] * 100
                accuracy = result_json["eval_accuracy"] * 100
                result.append(f"{f1:.2f}/{accuracy:.2f}")
            # 结果为 Pearson/Spearman corr.
            elif result_task_name == "stsb":
                pearson = result_json["eval_pearson"] * 100
                spearmanr = result_json["eval_spearmanr"] * 100
                result.append(f"{pearson:.2f}/{spearmanr:.2f}")
            # 结果为 Accuracy/F1
            elif result_task_name == "qqp":
                f1 = result_json["eval_f1"] * 100
                accuracy = result_json["eval_accuracy"] * 100
                result.append(f"{accuracy:.2f}/{f1:.2f}")
            # 结果为 Matched acc./Mismatched acc.
            elif result_task_name == "mnli":
                accuracy = result_json["eval_accuracy"] * 100
                accuracy_mm = result_json["eval_accuracy_mm"] * 100
                result.append(f"{accuracy:.2f}/{accuracy_mm:.2f}")
        results.append(result)
    return results


def ner_results():
    base_dir = os.path.join(script_path, "ner/result")
    result_dirs = sorted(os.listdir(base_dir))
    results = []
    for result_dir in result_dirs:
        result = [result_dir]
        result_path = os.path.join(base_dir, result_dir)
        for result_type in result_types:
            result_file = os.path.join(result_path, result_type, result_file_name)
            # 可能存在某项结果不存在
            if not os.path.exists(result_file):
                result.append("")
                continue
            result_json = json.load(open(result_file))
            # 结果为 Precision
            precision = result_json["eval_precision"] * 100
            result.append(f"{precision:.2f}")
        results.append(result)
    return results


def qa_results():
    base_dir = os.path.join(script_path, "qa/result")
    result_dirs = sorted(os.listdir(base_dir))
    results = []
    for result_dir in result_dirs:
        result = [result_dir]
        result_path = os.path.join(base_dir, result_dir)
        for result_type in result_types:
            result_file = os.path.join(result_path, result_type, result_file_name)
            # 可能存在某项结果不存在
            if not os.path.exists(result_file):
                result.append("")
                continue
            result_json = json.load(open(result_file))
            # 结果为 F1/Exact Match Accuracy, 这里本来就是0-100的数，所以不用缩放
            f1 = result_json["eval_f1"]
            exact = result_json["eval_exact"]
            result.append(f"{f1:.2f}/{exact:.2f}")
        results.append(result)
    return results


def main(downstream: str = None, glue_task_name: str = None) -> None:
    results = []
    for result_downstream in downstreams:
        if downstream != None and result_downstream != downstream:
            continue
        if result_downstream == "glue":
            results.extend(glue_results(glue_task_name))
        elif result_downstream == "ner":
            results.extend(ner_results())
        elif result_downstream == "qa":
            results.extend(qa_results())
    if len(results) == 0:
        print("还没有任何结果")
    else:
        headers = ["NAME", "CC", "BC", "CP", "BP"]
        print(tabulate(results, headers=headers, tablefmt="psql"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print results in tabular form')
    parser.add_argument(
        "-d", "--downstream", 
        type=str, 
        choices=downstreams, 
        default=None,
        help="Print the result of the specified downstream"
    )
    parser.add_argument(
        "-g", "--glue_task_name", 
        type=str, 
        choices=glue_task_names, 
        default=None,
        help="Print the result of the specified task"
    )
    args = parser.parse_args()
    main(args.downstream, args.glue_task_name)
