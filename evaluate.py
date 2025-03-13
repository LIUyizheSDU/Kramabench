"""This script can be used to evaluate the results of a system run.
The systems can be called using the run_pipeline function in system.api with the following run types:
- baseline: single model name (str)
- cross_validation: list of model names to cross validate, and a final model to merge the results
- reflection: one model for reflection, and another model to critic and decide to accept or reject the reflection
"""

from __future__ import print_function

import argparse
import json
import os
import traceback

import pandas as pd

from benchmark.metrics import metric_factory
from fixtures import system_selector


def build_prompt(details):
    prompt = "You will be given data sources and questions, please answer the questions based on the data sources. Plesae use <answer> to wrap the final answer, use <thinking> to wrap the thinking process if any."
    query = details.get("query", "")
    data_sources = details.get("data_sources", [])
    for i, source in enumerate(data_sources):
        prompt += f"\n\n<data_source_{i}>\n{source}\n</data_source_{i}>"
    prompt += f"\n\n<question>{query}</question>"
    return prompt


def parse_answer(answer):
    answer = answer.strip()
    # look for the the last <answer> and </answer>
    start = answer.rfind("<answer>")
    end = answer.rfind("</answer>")
    if start == -1 or end == -1:
        return answer
    return answer[start + len("<answer>") : end]


def compute_metrics(query, predicted, idx="", verbose=False):
    target = query["answer"]
    applicable_metrics = query["task_type"]["metrics"]

    results = []
    for metric_str in applicable_metrics:
        metric = metric_factory(metric_str)
        try:
            mean = metric(predicted, target)
        except Exception:
            print("Exception:", traceback.format_exc())
            if not verbose:
                print("On query:", idx)
            mean = 0

        dict_measures = {
            "metric": metric.name,
            "value": mean,
        }
        results.append(dict_measures)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sut", default="baseline-gpt-4o-mini", help="The system to benchmark"
    )
    parser.add_argument(
        "--subtasks",
        action="store_true",
        help="Whether the benchmark runs on subtasks and not on the parent-level tasks.",
    )
    parser.add_argument(
        "--workload",
        default="workload/jun-easy.json",
        help="The json file containing the input queries",
    )
    parser.add_argument(
        "--result",
        default="./results",
        help="The root path where the results of the pipelines are stored.",
    )
    parser.add_argument(
        "--use-cache",
        default=False,
        action="store_true",
        help="Whether to use cached results of the systems",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print filenames as they are processed",
    )

    args = parser.parse_args()
    sut = args.sut
    workload = args.workload
    RESULT_DIR = args.result
    run_subtasks = bool(args.subtasks)
    usecache = bool(args.use_cache)
    verbose = bool(args.verbose)

    subtasks_str = "_subtasks" if run_subtasks else ""
    workload_name = os.path.basename(workload)+subtasks_str
    result_file = f"{RESULT_DIR}/{sut}/{workload_name}_measures.csv"

    with open(workload) as f:
        queries = json.load(f)

    result_path = f"{RESULT_DIR}/{sut}/{workload_name}_results.json"
    sut_answers = {}

    if usecache:
        try:
            with open(result_path) as f:
                sut_answers = json.load(f)
        except Exception:
            print(
                f"Could not load the results of the system {sut} for workload {workload_name}. Processing."
            )

    if len(sut_answers) == 0:
        data_path = "data/TODO"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        system = system_selector(sut, verbose=verbose)
        system.process_dataset(data_path)
        for idx, query in enumerate(queries):
            if run_subtasks and len(query["subtasks"]) > 0:
                for jdx, subtask in enumerate(query["subtasks"]):
                    str_idx = f"{idx}_{jdx}"
                    if verbose:
                        print(f"Processing query {str_idx}")
                    prompt = build_prompt(subtask)
                    result = system.serve_query(prompt)
                    sut_answers[f"{idx}_{jdx}"] = parse_answer(result)
            else:  # if there are no subtasks, we just run the query
                if verbose:
                    print(f"Processing query {idx}")
                prompt = build_prompt(query)
                result = system.serve_query(prompt)
                sut_answers[str(idx)] = parse_answer(result)

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(sut_answers, f)

    workload_measures = []
    for idx, query in enumerate(queries):
        if run_subtasks and len(query["subtasks"]) > 0:
            for jdx, subtask in enumerate(query["subtasks"]):
                str_idx = f"{idx}_{jdx}"
                pred = sut_answers[str_idx]
                results = compute_metrics(subtask, pred, idx=str_idx, verbose=verbose)
                results = [r | {"workload": workload_name, "sut": sut} for r in results]
                workload_measures.extend(results)
        else:
            str_idx = str(idx)
            pred = sut_answers[str_idx]
            results = compute_metrics(query, pred, idx=str_idx, verbose=verbose)
            results = [r | {"workload": workload_name, "sut": sut} for r in results]
            workload_measures.extend(results)

    results_df = pd.DataFrame(workload_measures)
    results_df.to_csv(result_file, index=False)
    if verbose:
        print(results_df)

    # Logic to aggregate the results
    workload_results = []
    # group results_df by workload and metric
    for workload, group in results_df.groupby(["workload", "metric"]):
        workload, metric = workload
        mean = group["value"].mean()
        std = group["value"].std() if len(group) > 1 else 0
        workload_results.append(
            {
                "sut": sut,
                "workload": workload,
                "metric": metric,
                "value_mean": mean,
                "value_std": std,
                "value_support": len(group),
            }
        )

    aggregated_df = pd.DataFrame(workload_results)
    # read old aggregated results
    try:
        old_aggregated_df = pd.read_csv(f"{RESULT_DIR}/aggregated_results.csv")
    except Exception:
        old_aggregated_df = pd.DataFrame(
            columns=[
                "sut",
                "workload",
                "metric",
                "value_mean",
                "value_std",
                "value_support",
            ]
        )

    # remove old aggregated results with same sut and workload
    old_aggregated_df = old_aggregated_df[
        ~(
            (old_aggregated_df["workload"] == workload)
            & (old_aggregated_df["sut"] == sut)
        )
    ]
    aggregated_df = pd.concat([old_aggregated_df, aggregated_df])
    aggregated_df.to_csv(f"{RESULT_DIR}/aggregated_results.csv", index=False)

    print(aggregated_df[aggregated_df["workload"] == workload])


if __name__ == "__main__":
    main()
