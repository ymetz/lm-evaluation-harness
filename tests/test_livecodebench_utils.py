import json

from lm_eval.tasks.livecodebench import utils


def test_build_predictions_extracts_code_block():
    resps = [["```python\nprint('ok')\n```", "x = 1\nprint(x)"]]
    docs = [{}]
    output = utils.build_predictions(resps, docs)
    assert output == [["print('ok')", "x = 1\nprint(x)"]]


def test_pass_at_k_livecodebench_uses_estimator(monkeypatch):
    def fake_eval(**kwargs):
        del kwargs
        # one sample, two candidates: first passes, second fails
        return {0: [[True, True], [True, False]]}, {0: [{}, {}]}

    monkeypatch.setattr(utils, "evaluate_generations", fake_eval)

    references = [
        json.dumps(
            {
                "question_id": 1,
                "input_output": json.dumps(
                    {"inputs": ["1\n"], "outputs": ["1\n"], "fn_name": None}
                ),
            }
        )
    ]
    predictions = [["code_a", "code_b"]]
    metrics = utils.pass_at_k_livecodebench(
        references=references,
        predictions=predictions,
        k=[1, 2],
    )
    assert metrics["pass@1"] == 0.5
    assert metrics["pass@2"] == 1.0
