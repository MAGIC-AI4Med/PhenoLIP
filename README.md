<p align="center" width="100%">
</p>

<div id="top" align="center">

PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision–Language Pretraining
--------------------------------------------------------------------------------------------

<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://arxiv.org/pdf/2504.20930?"> 📑 Paper </a> |
<a href="https://github.com/MAGIC-AI4Med/ChestX-Reasoner"> 🐱 Github Repo </a> |
<a href="https://huggingface.co/byrLLCC/ChestX-Reasoner"> 🐱 PhenoLIP </a> |
  <a href="https://www.modelscope.cn/datasets/lcsjtu/PhenoBench"> 🐱 PhenoBench </a> |
</h4>

<!-- **Authors:** -->

_**Cheng Liang`<sup>`1,2 `</sup>`, Chaoyi Wu`<sup>`1,2 `</sup>`,Weike Zhao`<sup>`1,2 `</sup>`, Ya Zhang`<sup>`1,2`</sup>`, Yanfeng Wang`<sup>`1,2`</sup>`, Weidi Xie`<sup>`1,2`</sup>`**_

<!-- **Affiliations:** -->

_`<sup>`1`</sup>` Shanghai Jiao Tong University,
`<sup>`2`</sup>` Shanghai AI Laboratory._

</div>

The official codes for "PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision–Language Pretraining".

Note: We will make all the codes, data and model weights publicly available within one week.
## Data Processing

### Environment

You can install the code environment used for training our model.

```bash
conda create -n env_name python==3.10
conda activate env_name
pip3 install torch torchvision

```

* Python: Version >= 3.9
* CUDA: Version >= 12.1
* VLLM: Version >= 0.7

## Evaluation

### Benchmark Data

In `eval/data`, we present our benchmark construction code and our data.

### Evaluation

We provide:

1. The evaluation code on both reasoning and accuracy in `eval/`
2. The baseline inference code in `eval/inference`
3. The evaluation results on both reasoning and accuracy of all baselines in `eval/res`

## Training Data

## Citation

If you find this work is relevant with your research or applications, please feel free to cite our work!

```
@article{
    xxx
}
```
