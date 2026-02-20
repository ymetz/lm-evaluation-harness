from lm_eval.tasks.code_execution import compute_pass_at_k


def test_compute_pass_at_k_subprocess_perfect():
    references = ["assert add(2, 3) == 5"]
    predictions = [["def add(a, b):\n    return a + b"]]
    metrics = compute_pass_at_k(
        references=references,
        predictions=predictions,
        k=[1],
        backend="subprocess",
        timeout_sec=2.0,
        n_workers=1,
    )
    assert metrics["pass@1"] == 1.0


def test_compute_pass_at_k_subprocess_imperfect():
    references = ["assert add(2, 3) == 5"]
    predictions = [["def add(a, b):\n    return a - b"]]
    metrics = compute_pass_at_k(
        references=references,
        predictions=predictions,
        k=[1],
        backend="subprocess",
        timeout_sec=2.0,
        n_workers=1,
    )
    assert metrics["pass@1"] == 0.0


def test_compute_pass_at_k_subprocess_k_greater_than_n():
    references = ["assert add(2, 3) == 5"]
    predictions = [["def add(a, b):\n    return a + b"]]
    metrics = compute_pass_at_k(
        references=references,
        predictions=predictions,
        k=[10],
        backend="subprocess",
        timeout_sec=2.0,
        n_workers=1,
    )
    assert metrics["pass@10"] == 1.0
