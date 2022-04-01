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

def glue_results(task_name: str = None):
    base_dir = "glue/result"
    result_dirs = os.listdir(base_dir)
    result_dirs.sort()
    results = []
    for result_dir in result_dirs:
        result_task_name = result_dir.split("-")[0]
        if task_name != None and result_task_name != task_name:
            continue
        result = [result_dir]
        result_path = os.path.join(base_dir, result_dir)
        for result_type in result_types:
            fp = open(os.path.join(result_path, result_type, result_file_name))
            result_json = json.load(fp)
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
    base_dir = "ner/result"
    result_dirs = os.listdir(base_dir)
    result_dirs.sort()
    results = []
    for result_dir in result_dirs:
        result = [result_dir]
        result_path = os.path.join(base_dir, result_dir)
        for result_type in result_types:
            fp = open(os.path.join(result_path, result_type, result_file_name))
            result_json = json.load(fp)
            # 结果为 Accuracy/F1/Recall/Precision
            accuracy = result_json["eval_accuracy"] * 100
            f1 = result_json["eval_f1"] * 100
            recall = result_json["eval_recall"] * 100
            precision = result_json["eval_precision"] * 100
            result.append(f"{accuracy:.2f}/{f1:.2f}/{recall:.2f}/{precision:.2f}/")
        results.append(result)
    return results


def qa_results():
    results = []
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
