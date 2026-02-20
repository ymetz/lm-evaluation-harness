# LiveCodeBench

## Paper
LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code  
https://arxiv.org/abs/2403.07974

LiveCodeBench provides coding questions sourced from recent contests to reduce contamination risk. Problems include public and private test cases.

Homepage: https://livecodebench.github.io/  
Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite

## Citation
```bibtex
@misc{jain2024livecodebench,
      title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
      author={Naman Jain and King Han and Alex Gu and Wen-Ding Li and Fanjia Yan and Tianjun Zhang and Sida Wang and Armando Solar-Lezama and Koushik Sen and Ion Stoica},
      year={2024},
      eprint={2403.07974},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Tasks
- `livecodebench`: release_v3, sampling setup (`repeats: 10`, pass@1/5/10).
- `livecodebench_1`: deterministic pass@1 setup (`repeats: 1`).
- `livecodebench_all`: group for both variants.

## Notes
- Uses local test execution via `testing_utils.py` (copied from LiveCodeBench reference runner).
- Task is marked `unsafe_code: true`; run with `--confirm_run_unsafe_code`.
