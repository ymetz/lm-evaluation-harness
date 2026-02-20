import json

from lm_eval.tasks.bigcodebench import utils


def test_build_predictions_complete_combines_prompt_and_code():
    resps = [["```python\nreturn x + 1\n```"]]
    docs = [{"complete_prompt": "def f(x):\n"}]
    output = utils.build_predictions_complete(resps, docs)
    assert output == [["def f(x):\nreturn x + 1"]]


def test_pass_at_k_bcb_uses_remote_results(monkeypatch):
    def fake_remote_call(**kwargs):
        del kwargs
        return [{"status": "pass"}, {"status": "fail"}]

    monkeypatch.setattr(utils, "call_oe_eval_bcb_client", fake_remote_call)

    references = [
        json.dumps(
            {
                "test": "assert f(1) == 2",
                "entry_point": "f",
                "code_prompt": "def f(x):\n",
                "task_id": "task_1",
            }
        )
    ]
    predictions = [["return x + 1", "return x - 1"]]
    metrics = utils.pass_at_k_bcb(references=references, predictions=predictions, k=[1, 2])

    assert metrics["pass@1"] == 0.5
    assert metrics["pass@2"] == 1.0
