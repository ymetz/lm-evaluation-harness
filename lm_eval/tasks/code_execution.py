from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Sequence


def _as_k_list(k: int | Sequence[int] | None) -> list[int]:
    if k is None:
        raise ValueError("k must be provided for pass@k computation.")
    if isinstance(k, int):
        return [k]
    return [int(ki) for ki in k]


def _normalize_predictions(
    references: Sequence[str], predictions: Sequence[Sequence[str] | str]
) -> list[list[str]]:
    if len(references) != len(predictions):
        raise ValueError(
            "references and predictions must have the same length. "
            f"Got {len(references)} references and {len(predictions)} predictions."
        )

    normalized: list[list[str]] = []
    for pred in predictions:
        if isinstance(pred, str):
            normalized.append([pred])
        else:
            normalized.append([str(p) for p in pred])
    return normalized


def _extract_metric_dict(result: Any) -> dict[str, float]:
    if isinstance(result, tuple):
        if not result:
            raise ValueError("Received an empty tuple from backend metric computation.")
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError(
            "Expected metric backend to return a dict (or tuple containing dict). "
            f"Got {type(result)}."
        )
    return {str(k): float(v) for k, v in result.items()}


@lru_cache(maxsize=1)
def _get_hf_code_eval_metric():
    import evaluate as hf_evaluate

    return hf_evaluate.load("code_eval")


def _compute_hf_code_eval(
    references: list[str], predictions: list[list[str]], k_list: list[int]
) -> dict[str, float]:
    metric = _get_hf_code_eval_metric()
    result = metric.compute(references=references, predictions=predictions, k=k_list)
    return _extract_metric_dict(result)


def _run_subprocess_case(candidate: str, reference: str, timeout_sec: float) -> bool:
    # `unsafe_code: true` tasks already require explicit user opt-in. We execute in
    # a subprocess and treat any non-zero exit, timeout, or runtime error as failure.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as temp_file:
        temp_file.write(candidate)
        temp_file.write("\n\n")
        temp_file.write(reference)
        temp_file.write("\n")
        temp_path = temp_file.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=False,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _compute_subprocess_code_eval(
    references: list[str],
    predictions: list[list[str]],
    k_list: list[int],
    timeout_sec: float = 10.0,
    n_workers: int = 8,
) -> dict[str, float]:
    passed: list[list[bool]] = [[False] * len(preds) for preds in predictions]

    jobs: list[tuple[int, int, str, str]] = []
    for doc_idx, (reference, candidates) in enumerate(zip(references, predictions)):
        for pred_idx, candidate in enumerate(candidates):
            jobs.append((doc_idx, pred_idx, candidate, reference))

    max_workers = max(1, int(n_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_subprocess_case, candidate, reference, float(timeout_sec)
            ): (doc_idx, pred_idx)
            for doc_idx, pred_idx, candidate, reference in jobs
        }
        for future in as_completed(futures):
            doc_idx, pred_idx = futures[future]
            try:
                passed[doc_idx][pred_idx] = bool(future.result())
            except Exception:
                passed[doc_idx][pred_idx] = False

    metric: dict[str, float] = {}
    for k_val in k_list:
        doc_scores = []
        for row in passed:
            n = len(row)
            c = sum(1 for ok in row if ok)
            doc_scores.append(_estimate_pass_at_k(n=n, c=c, k=k_val))
        metric[f"pass@{k_val}"] = (
            float(sum(doc_scores) / len(doc_scores)) if doc_scores else 0.0
        )
    return metric


def _compute_remote_code_eval(
    references: list[str],
    predictions: list[list[str]],
    k_list: list[int],
    api_url: str,
    api_key: str | None = None,
    timeout_sec: float = 60.0,
    request_timeout_sec: float = 120.0,
) -> dict[str, float]:
    payload = {
        "references": references,
        "predictions": predictions,
        "k": k_list,
        "timeout_sec": float(timeout_sec),
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=float(request_timeout_sec)) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Remote code execution backend returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Remote code execution backend request failed: {exc.reason}"
        ) from exc

    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError(
            f"Remote backend response must be a JSON object; got {type(data)}."
        )

    if "metrics" in data and isinstance(data["metrics"], dict):
        return {str(k): float(v) for k, v in data["metrics"].items()}

    if all(f"pass@{k_val}" in data for k_val in k_list):
        return {f"pass@{k_val}": float(data[f"pass@{k_val}"]) for k_val in k_list}

    if "passed" in data and isinstance(data["passed"], list):
        passed = data["passed"]
        if len(passed) != len(predictions):
            raise ValueError(
                "Remote backend returned a `passed` matrix with unexpected row count. "
                f"Expected {len(predictions)}, got {len(passed)}."
            )
        metric: dict[str, float] = {}
        for k_val in k_list:
            doc_scores = []
            for row, preds in zip(passed, predictions):
                if not isinstance(row, list):
                    raise ValueError("Each row in remote `passed` must be a list.")
                if len(row) != len(preds):
                    raise ValueError(
                        "Remote backend returned a `passed` row with unexpected size. "
                        f"Expected {len(preds)}, got {len(row)}."
                    )
                n = len(row)
                c = sum(1 for ok in row if bool(ok))
                doc_scores.append(_estimate_pass_at_k(n=n, c=c, k=k_val))
            metric[f"pass@{k_val}"] = (
                float(sum(doc_scores) / len(doc_scores)) if doc_scores else 0.0
            )
        return metric

    raise ValueError(
        "Remote backend response must provide either "
        "`metrics`, direct `pass@k` keys, or a `passed` matrix."
    )


def compute_pass_at_k(
    references: Sequence[str],
    predictions: Sequence[Sequence[str] | str],
    k: int | Sequence[int],
    backend: str | None = None,
    timeout_sec: float = 10.0,
    n_workers: int = 8,
    api_url: str | None = None,
    api_key: str | None = None,
    request_timeout_sec: float = 120.0,
) -> dict[str, float]:
    """Compute pass@k using one of three backends.

    Supported `backend` values:
    - `"hf"` (default): use `evaluate.load("code_eval")`
    - `"subprocess"`: run candidates against tests locally in subprocesses
    - `"remote"`: send evaluation requests to a REST endpoint

    Environment variable fallbacks:
    - `LMEVAL_CODE_EXEC_BACKEND`
    - `LMEVAL_CODE_EXEC_API_URL`
    - `LMEVAL_CODE_EXEC_API_KEY`
    """
    k_list = _as_k_list(k)
    references_norm = [str(r) for r in references]
    predictions_norm = _normalize_predictions(references_norm, predictions)

    backend_name = (
        backend or os.getenv("LMEVAL_CODE_EXEC_BACKEND") or "hf"
    ).strip().lower()

    if backend_name == "hf":
        return _compute_hf_code_eval(references_norm, predictions_norm, k_list)

    if backend_name == "subprocess":
        return _compute_subprocess_code_eval(
            references=references_norm,
            predictions=predictions_norm,
            k_list=k_list,
            timeout_sec=timeout_sec,
            n_workers=n_workers,
        )

    if backend_name == "remote":
        resolved_url = api_url or os.getenv("LMEVAL_CODE_EXEC_API_URL")
        resolved_key = api_key or os.getenv("LMEVAL_CODE_EXEC_API_KEY")
        if not resolved_url:
            raise ValueError(
                "Remote backend selected but no api_url provided. Set `api_url` or "
                "LMEVAL_CODE_EXEC_API_URL."
            )
        return _compute_remote_code_eval(
            references=references_norm,
            predictions=predictions_norm,
            k_list=k_list,
            api_url=resolved_url,
            api_key=resolved_key,
            timeout_sec=timeout_sec,
            request_timeout_sec=request_timeout_sec,
        )

    raise ValueError(
        f"Unknown code execution backend '{backend_name}'. "
        "Supported backends are: hf, subprocess, remote."
    )
