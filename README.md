<p align="center" width="100%">
</p>

<div id="top" align="center">

PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision–Language Pretraining
--------------------------------------------------------------------------------------------

<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://magic-ai4med.github.io/phenolip-webpage/"> 🌐 Project Page </a> |
<a href="https://arxiv.org/pdf/2504.20930?"> 📑 Paper </a> |
<a href="https://github.com/MAGIC-AI4Med/PhenoLIP"> <img src="https://img.shields.io/badge/GitHub-black?logo=github" alt="GitHub"> Github Repo </a> |
<a href="https://www.modelscope.cn/models/lcsjtu/PhenoLIP"> <img src="https://img.shields.io/badge/ModelScope-blue?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMiA3TDEyIDEyTDIyIDdMMTIgMloiIGZpbGw9IndoaXRlIi8+CjxwYXRoIGQ9Ik0yIDdWMTdMMTIgMjJWMTJMMiA3WiIgZmlsbD0id2hpdGUiLz4KPHBhdGggZD0iTTEyIDEyVjIyTDIyIDE3VjdMMTIgMTJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4=" alt="ModelScope"> PhenoLIP </a> |
  <a href="https://www.modelscope.cn/datasets/lcsjtu/PhenoBench"> <img src="https://img.shields.io/badge/ModelScope-blue?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMiA3TDEyIDEyTDIyIDdMMTIgMloiIGZpbGw9IndoaXRlIi8+CjxwYXRoIGQ9Ik0yIDdWMTdMMTIgMjJWMTJMMiA3WiIgZmlsbD0id2hpdGUiLz4KPHBhdGggZD0iTTEyIDEyVjIyTDIyIDE3VjdMMTIgMTJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4=" alt="ModelScope"> PhenoBench </a> |
</h4>

<!-- **Authors:** -->

<em><strong>Cheng Liang<sup>1,2</sup>, Chaoyi Wu<sup>1</sup>, Weike Zhao<sup>1,2</sup>, Ya Zhang<sup>1,2</sup>, Yanfeng Wang<sup>1,2</sup>, Weidi Xie<sup>1,2</sup></strong></em>

<em><sup>1</sup> Shanghai Jiao Tong University,
<sup>2</sup> Shanghai AI Laboratory.</em>

</div>

The official codes for "PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision–Language Pretraining".

## Project Structure

- **Alignment/**: Sub-figure level image-text alignment
- **Augment/**: Sub-figure level caption augmentation
- **Cls/**: Image classification
- **Cluster/**: Data clustering
- **Detection/**: Sub-figure detection
- **Filter/**: Article filtering from PubMed
- **OCR/**: Image content recognition
- **eval/**: Benchmark construction and evaluation code

### Environment

You can install the code environment used for training our model.

```bash
conda create -n phenolip python==3.10
conda activate phenolip
pip install torch==2.6.0 torchvision==0.21.0

```

* Python: Version >= 3.10
* CUDA: Version >= 12.1
* VLLM: Version >= 0.7


## Citation

If you find this work is relevant with your research or applications, please feel free to cite our work!

```
@misc{liang2026phenolipintegratingphenotypeontology,
      title={PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision-Language Pretraining}, 
      author={Cheng Liang and Chaoyi Wu and Weike Zhao and Ya Zhang and Yanfeng Wang and Weidi Xie},
      year={2026},
      eprint={2602.06184},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.06184}, 
}
```
