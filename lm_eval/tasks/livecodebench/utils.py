import json
import math
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Sequence

from lm_eval.tasks.livecodebench.testing_utils import decode_private_test_cases, run_test


def _process_single_doc(doc: dict) -> dict:
    out_doc = dict(doc)

    problem_statement = str(doc["question_content"])
    has_starter_code = str(doc.get("starter_code", "")).strip()
    if has_starter_code:
        format_instruction = (
            "You will use the following starter code to write the solution to the "
            "problem and enclose your code within delimiters."
        )
        format_instruction += f"\n```python\n{doc['starter_code']}\n```"
    else:
        format_instruction = (
            "Read the inputs from stdin solve the problem and write the answer to "
            "stdout (do not directly test on the sample inputs). Enclose your code "
            "within delimiters as follows. Ensure that when the python program runs, "
            "it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```"
        )

    system_message = (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program that "
        "matches the specification and passes all tests."
    )
    question_template = f"### Question:\n{problem_statement}\n\n"
    format_template = f"### Format: {format_instruction}\n"
    answer_template = "### Answer: (use the provided format with backticks)\n\n"
    out_doc["query"] = system_message + "\n\n" + question_template + format_template + answer_template

    public_test_cases = json.loads(doc["public_test_cases"]) if doc.get("public_test_cases") else []
    private_test_cases = decode_private_test_cases(doc.get("private_test_cases"))
    all_test_cases = public_test_cases + private_test_cases

    metadata_raw = doc.get("metadata")
    if isinstance(metadata_raw, str) and metadata_raw.strip():
        metadata = json.loads(metadata_raw)
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        metadata = {}

    out_doc["metadata"] = metadata
    out_doc["num_public_tests"] = len(public_test_cases)
    out_doc["num_private_tests"] = len(private_test_cases)
    out_doc["num_total_tests"] = len(all_test_cases)

    fn_name = metadata.get("func_name")
    input_output = {
        "inputs": [t["input"] for t in all_test_cases],
        "outputs": [t["output"] for t in all_test_cases],
        "fn_name": fn_name,
    }
    out_doc["input_output"] = json.dumps(input_output, ensure_ascii=False)
    out_doc["eval_reference"] = json.dumps(
        {
            "question_id": out_doc.get("question_id"),
            "input_output": out_doc["input_output"],
        },
        ensure_ascii=False,
    )
    return out_doc


def process_docs(dataset):
    return dataset.map(_process_single_doc)


def doc_to_target(doc: dict) -> str:
    return doc["eval_reference"]


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    del docs
    return [[_extract_code(r) for r in resp] for resp in resps]


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    run_test_result = run_test(sample, test=generation, debug=debug, timeout=int(timeout))
    if run_test_result:
        res, metadata = run_test_result
    else:
        in_outs = json.loads(sample["input_output"])
        num_tests = len(in_outs["inputs"])
        res = [-4] * num_tests
        metadata = {
            "error_code": -4,
            "error_message": "Function not found in generated code or compilation error.",
        }
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=False):
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    process = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    process.start()
    process.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if process.is_alive():
        process.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        metadata_list = [{"error_code": -1, "error_message": "Global timeout"}]
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    problem_generations = args[0]
    sample = args[1]
    debug = args[2]
    timeout = args[3]

    results = []
    metadata = []
    for generation in problem_generations:
        curr_res = [-2]
        curr_metadata: dict[str, Any] = {
            "error_code": -5,
            "error_message": "Unknown",
        }
        try:
            curr_res, curr_metadata = check_correctness(
                sample, generation, timeout=timeout, debug=debug
            )
            fixed = []
            for item in curr_res:
                if hasattr(item, "item"):
                    item = item.item(0)
                if type(item).__name__ == "bool_":
                    item = bool(item)
                fixed.append(item)
            curr_res = fixed
        except Exception as exc:
            curr_metadata = {
                "error": repr(exc),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            results.append(curr_res)
            metadata.append(curr_metadata)
    return results, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout: float = 6.0,
):
    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
        futures = {
            executor.submit(evaluate_generations_by_problem, arg): index
            for arg, index in inputs
        }
        results = {}
        metadata = {}
        for future in as_completed(futures):
            index = futures[future]
            results[index], metadata[index] = future.result()
    return results, metadata


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _as_k_list(k: int | Sequence[int] | None) -> list[int]:
    if k is None:
        return [1]
    if isinstance(k, int):
        return [k]
    return [int(ki) for ki in k]


def pass_at_k_livecodebench(
    references: list[str],
    predictions: list[list[str]] | list[str],
    k: int | Sequence[int] | None = None,
    timeout: float = 6.0,
    n_exe_workers: int = 20,
    debug: bool = False,
    **_,
) -> dict[str, float]:
    if predictions and isinstance(predictions[0], str):
        predictions = [[pred] for pred in predictions]  # type: ignore[assignment]

    samples_list = [json.loads(ref) for ref in references]
    samples_list = [{"input_output": sample["input_output"]} for sample in samples_list]
    generations_list = [[str(p) for p in preds] for preds in predictions]
    k_list = _as_k_list(k)

    results, _metadata = evaluate_generations(
        samples_list=samples_list,
        generations_list=generations_list,
        timeout=timeout,
        num_process_evaluate=n_exe_workers,
        debug=debug,
    )

    metrics = {f"pass@{k_val}": [] for k_val in k_list}
    for sample_index, candidates in enumerate(generations_list):
        generation_results = results.get(sample_index, [])
        passed_count = 0
        for i in range(len(candidates)):
            if i >= len(generation_results):
                continue
            passed = all(item is True for item in generation_results[i])
            if passed:
                passed_count += 1
        n = len(candidates)
        for k_val in k_list:
            metrics[f"pass@{k_val}"].append(_estimate_pass_at_k(n=n, c=passed_count, k=k_val))

    return {
        metric_name: (sum(vals) / len(vals) if vals else 0.0)
        for metric_name, vals in metrics.items()
    }
