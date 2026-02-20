import re
from typing import Union

from lm_eval.tasks.code_execution import compute_pass_at_k


def pass_at_1(
    references: Union[str, list[str]],
    predictions: Union[str, list[list[str]]],
    backend: str | None = None,
    timeout_sec: float = 10.0,
    n_workers: int = 8,
    api_url: str | None = None,
    api_key: str | None = None,
    request_timeout_sec: float = 120.0,
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [[predictions]]
    elif isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return compute_pass_at_k(
        references=references,
        predictions=predictions,
        k=1,
        backend=backend,
        timeout_sec=timeout_sec,
        n_workers=n_workers,
        api_url=api_url,
        api_key=api_key,
        request_timeout_sec=request_timeout_sec,
    )["pass@1"]


def clean_text(text: str) -> str:
    text = re.sub(r"\n(▁+)", lambda m: "\n" + " " * len(m.group(1)), text)

    # strip leading/trailing whitespace
    text = text.strip()
    # remove stray leading space before def
    text = re.sub(r"^\s*def", "def", text, flags=re.MULTILINE)
    return text


def extract_code_blocks(text: str) -> str:
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    # (+ ```) as we add the opening "```python" to the gen_prefix
    matches = re.findall(pattern, r"```" + text, re.DOTALL)
    # if no matches, try to match ```...``` blocks (after removing the language)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if not matches:
        return ""
    else:
        return clean_text(matches[0])


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n    res = tuple(set(test_tup1) & set(test_tup2))\r\n    return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n    largest_nums = hq.nlargest(n, nums)\r\n    return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
