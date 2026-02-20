# BigCodeBench

## Paper
BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions  
https://arxiv.org/pdf/2406.15877

BigCodeBench evaluates practical function-level Python code generation with more complex instructions and broader API usage than HumanEval-style tasks.

Code: https://github.com/bigcode-project/bigcodebench

## Citation
```bibtex
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Minh Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
```

## Tasks
- `bigcodebench`: v0.1.2 complete-prompt style, pass@1 default.
- `bigcodebench_instruct`: v0.1.2 instruct-prompt style.
- `bigcodebench_5`: complete-prompt style with 5 samples (pass@1 and pass@5).
- `bigcodebench_all`: task group including all variants above.

## Remote Execution
This implementation uses remote BigCodeBench execution (OE-style evaluator endpoint) from `utils.pass_at_k_bcb`.

Optional environment variables:
- `LMEVAL_BCB_REMOTE_API`: override evaluator URL.
  - Default: `https://jjyang7-oe-eval-bcb-lite-evaluator.hf.space/evaluate/`
- `LMEVAL_BCB_REMOTE_API_KEY`: optional bearer token.

The task is marked `unsafe_code: true`, so run with:
```bash
lm_eval --tasks bigcodebench --confirm_run_unsafe_code ...
```
