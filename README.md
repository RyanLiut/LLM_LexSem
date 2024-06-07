# Fantastic Semantics and Where to Find Them: Investigating Which Layers of Generative LLMs Reflect Lexical Semantics

This repository is for the papaer (published in Findings: ACL 2024): [Fantastic Semantics and Where to Find Them: Investigating Which Layers of Generative LLMs Reflect Lexical Semantics](https://arxiv.org/abs/2403.01509)

## Code Execution

The data `dir` includes two datasets: [WiC](https://pilehvar.github.io/wic/) and [RAW-C](https://github.com/seantrott/raw-c). We mainly evaluate LLMs on the first dataset.

Llama models should be downloaded into the root dir. To download these models, please refer to the [official repo](https://github.com/meta-llama/llama).

Run the `get_thresholds.py` before run the `main.py`. (We add some thresholds used in our experiments in the `data` directory.)

## Citation

```
@article{liu2024fantastic,
  title={Fantastic Semantics and Where to Find Them: Investigating Which Layers of Generative LLMs Reflect Lexical Semantics},
  author={Liu, Zhu and Kong, Cunliang and Liu, Ying and Sun, Maosong},
  journal={arXiv preprint arXiv:2403.01509},
  year={2024}
}
```