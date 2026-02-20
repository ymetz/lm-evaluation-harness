import re

from lm_eval.tasks.code_execution import compute_pass_at_k


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: int | list[int] | None = None,
    backend: str | None = None,
    timeout_sec: float = 10.0,
    n_workers: int = 8,
    api_url: str | None = None,
    api_key: str | None = None,
    request_timeout_sec: float = 120.0,
):
    assert k is not None
    return compute_pass_at_k(
        references=references,
        predictions=predictions,
        k=k,
        backend=backend,
        timeout_sec=timeout_sec,
        n_workers=n_workers,
        api_url=api_url,
        api_key=api_key,
        request_timeout_sec=request_timeout_sec,
    )


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + clean_text(r.replace("```python\n", "")) for r in resp]
        for resp, doc in zip(resps, docs)
    ]


def clean_text(text: str) -> str:
    return re.sub(r"\n(▁+)", lambda m: "\n" + " " * len(m.group(1)), text)


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"]
            + (clean_text(r) if r.find("```") == -1 else clean_text(r[: r.find("```")]))
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
