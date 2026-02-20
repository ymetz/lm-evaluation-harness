import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Sequence

import datasets


INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
DEFAULT_REMOTE_EXEC_API = (
    "https://jjyang7-oe-eval-bcb-lite-evaluator.hf.space/evaluate/"
)


def load_bigcodebench_dataset(
    dataset_path: str = "bigcode/bigcodebench",
    dataset_name: str | None = None,
    split: str = "v0.1.2",
    **kwargs,
):
    dataset = datasets.load_dataset(
        path=dataset_path,
        name=dataset_name,
        split=split,
        **kwargs,
    )
    return {"test": dataset}


def _process_doc(doc: dict, prompt_field: str) -> dict:
    if prompt_field not in doc:
        raise KeyError(
            f"Expected prompt field '{prompt_field}' in BigCodeBench document."
        )
    out_doc = dict(doc)
    out_doc["prompt"] = doc[prompt_field]
    out_doc["eval_reference"] = json.dumps(
        {
            "test": doc["test"],
            "entry_point": doc["entry_point"],
            "code_prompt": doc.get("code_prompt", ""),
            "task_id": doc.get("task_id"),
        },
        ensure_ascii=False,
    )
    return out_doc


def process_docs_complete(dataset):
    return dataset.map(lambda doc: _process_doc(doc, "complete_prompt"))


def process_docs_instruct(dataset):
    return dataset.map(lambda doc: _process_doc(doc, "instruct_prompt"))


def doc_to_text_complete(doc: dict) -> str:
    return f"{INSTRUCTION_PREFIX}\n```\n{doc['prompt'].strip()}\n"


def doc_to_text_instruct(doc: dict) -> str:
    return f"{INSTRUCTION_PREFIX}\n{doc['prompt'].strip()}\n"


def doc_to_target(doc: dict) -> str:
    return doc["eval_reference"]


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_code_block(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _sanitize_candidate(candidate: str) -> str:
    code = _extract_code_block(candidate)
    code = code.replace("\r\n", "\n")
    code = code.rstrip()
    if code.startswith("python\n"):
        code = code[len("python\n") :]
    return code


def _join_prompt_and_completion(prompt: str, completion: str) -> str:
    if not prompt:
        return completion
    if not completion:
        return prompt
    if prompt.endswith("\n"):
        return prompt + completion
    return prompt + "\n" + completion


def build_predictions_complete(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    output = []
    for resp, doc in zip(resps, docs):
        prompt = doc.get("complete_prompt", "")
        output.append(
            [_join_prompt_and_completion(prompt, _sanitize_candidate(r)) for r in resp]
        )
    return output


def build_predictions_instruct(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[_sanitize_candidate(r) for r in resp] for resp in resps]


def _status_is_pass(item: dict) -> bool:
    if "passed" in item:
        return bool(item["passed"])
    if "status" in item:
        return str(item["status"]).lower() == "pass"
    return False


def _flatten_eval_payload(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("eval"), dict):
        flattened = []
        for reps in payload["eval"].values():
            if isinstance(reps, list):
                flattened.extend(reps)
        return flattened
    raise ValueError("Unexpected response format from remote BigCodeBench evaluator.")


def call_oe_eval_bcb_client(
    samples_data: list[dict],
    calibrate: bool = True,
    parallel: int = -1,
    min_time_limit: float = 1.0,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    no_gt: bool = True,
    remote_execute_api: str | None = None,
    api_key: str | None = None,
    retries: int = 3,
    timeout_sec: float = 3600.0,
) -> list[dict]:
    remote_execute_api = (
        remote_execute_api
        or os.getenv("LMEVAL_BCB_REMOTE_API")
        or DEFAULT_REMOTE_EXEC_API
    )
    api_key = api_key or os.getenv("LMEVAL_BCB_REMOTE_API_KEY")

    params = {
        "calibrate": str(bool(calibrate)).lower(),
        "parallel": int(parallel),
        "min_time_limit": float(min_time_limit),
        "max_as_limit": int(max_as_limit),
        "max_data_limit": int(max_data_limit),
        "max_stack_limit": int(max_stack_limit),
        "no_gt": str(bool(no_gt)).lower(),
    }
    url = f"{remote_execute_api}?{urllib.parse.urlencode(params)}"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload_bytes = json.dumps(samples_data).encode("utf-8")
    sleep_sec = 5.0
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url=url,
                data=payload_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
            return _flatten_eval_payload(json.loads(raw))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(sleep_sec)
            sleep_sec = min(60.0, sleep_sec * 2)
    raise RuntimeError(
        "Failed to call remote BigCodeBench evaluator after retries."
    ) from last_error


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _as_k_list(
    k: int | Sequence[int] | None = None,
    pass_at_ks: Sequence[int] | None = None,
) -> list[int]:
    if k is None:
        k = pass_at_ks if pass_at_ks is not None else [1]
    if isinstance(k, int):
        return [k]
    return [int(ki) for ki in k]


def _parse_reference(reference: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(reference, dict):
        return reference
    return json.loads(reference)


def pass_at_k_bcb(
    references: list[str],
    predictions: list[list[str]] | list[str],
    k: int | Sequence[int] | None = None,
    pass_at_ks: Sequence[int] | None = None,
    calibrate_code: bool = True,
    parallel: int = -1,
    min_time_limit: float = 1.0,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    no_gt: bool = True,
    remote_execute_api: str | None = None,
    api_key: str | None = None,
    rich_exec_info: bool = True,
    **_,
) -> dict[str, float]:
    del rich_exec_info

    k_list = _as_k_list(k=k, pass_at_ks=pass_at_ks)
    if predictions and isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]  # type: ignore[assignment]

    metric_values = {f"pass@{k_val}": [] for k_val in k_list}
    for reference, candidates in zip(references, predictions):
        reference_data = _parse_reference(reference)
        candidate_list = [str(c) for c in candidates]
        if not candidate_list:
            for k_val in k_list:
                metric_values[f"pass@{k_val}"].append(0.0)
            continue

        samples_data = []
        for completion in candidate_list:
            samples_data.append(
                {
                    "solution": completion,
                    "test": reference_data["test"],
                    "entry_point": reference_data["entry_point"],
                    "code_prompt": reference_data.get("code_prompt", ""),
                    "task_id": reference_data.get("task_id"),
                }
            )

        eval_results = call_oe_eval_bcb_client(
            samples_data=samples_data,
            calibrate=calibrate_code,
            parallel=parallel,
            min_time_limit=min_time_limit,
            max_as_limit=max_as_limit,
            max_data_limit=max_data_limit,
            max_stack_limit=max_stack_limit,
            no_gt=no_gt,
            remote_execute_api=remote_execute_api,
            api_key=api_key,
        )

        statuses = [_status_is_pass(item) for item in eval_results[: len(candidate_list)]]
        if len(statuses) < len(candidate_list):
            statuses.extend([False] * (len(candidate_list) - len(statuses)))
        c = sum(1 for ok in statuses if ok)
        n = len(candidate_list)
        for k_val in k_list:
            metric_values[f"pass@{k_val}"].append(_estimate_pass_at_k(n=n, c=c, k=k_val))

    return {
        metric_name: (sum(vals) / len(vals) if vals else 0.0)
        for metric_name, vals in metric_values.items()
    }
