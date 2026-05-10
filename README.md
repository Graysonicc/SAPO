<h1 align="left">[ICML 2026] Segment-Aligned Policy Optimization for Multi-Modal Reasoning</h1>

<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/Repo-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/Graysonicc/SAPO/tree/main)
[![arXiv](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=arXiv&logoColor=white&labelColor)](http://arxiv.org/abs/2605.01327)

</div>

---
## Installation

You can follow the instruction from [VeRL](https://verl.readthedocs.io/en/latest/start/install.html), the version is v0.4.1

Relevant package versions include:
```
conda create -n sapo python==3.10

verl == 0.4.1
vllm == 0.8.4
trl == 0.14.0
ray == 2.46.0
torch == 2.6.0+cu124
transformers == 4.57.1
flash-attn == 2.7.4.post1
```

## Training
Train the SAPO with script:
```
cd SAPO
bash examples/ppo_trainer/run_sapo.sh
```

## Citation
If you find this code useful for your research, please cite the following paper.

```bibtex
@article{gao2026segment,
  title={Segment-Aligned Policy Optimization for Multi-Modal Reasoning},
  author={Gao, Lei and Li, Zhuoming and Jia, Mengxi and Yuan, Jiakang and Sun, Hongbo and Sun, Hao and Li, Xuelong},
  journal={arXiv preprint arXiv:2605.01327},
  year={2026}
}
```