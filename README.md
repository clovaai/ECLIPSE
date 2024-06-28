# ECLIPSE (CVPR 2024)

**ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning** <br />
[Beomyoung Kim](https://beomyoung-kim.github.io/)<sup>1,2</sup>, [Joonsang Yu]()<sup>1</sup>, [Sung Ju Hwang](http://www.sungjuhwang.com/)<sup>2</sup><br>

<sup>1</sup> <sub>NAVER Cloud, ImageVision</sub><br />
<sup>2</sup> <sub>KAIST</sub><br />

[![](https://img.shields.io/badge/CVPR-2024-blue)](https://cvpr.thecvf.com/Conferences/2024)
[![Paper](https://img.shields.io/badge/Paper-arxiv.2403.20126-red)](https://arxiv.org/abs/2403.20126)
[![](https://img.shields.io/badge/YouTube-red)](https://youtu.be/L3qaH-mluiE?si=LrDmKazNYz2tBHp7)

</div>

![demo image](https://github.com/clovaai/ECLIPSE/releases/download/assets/eclipse.gif)


## Introduction

Panoptic segmentation, combining semantic and instance segmentation, stands as a cutting-edge computer vision task. Despite recent progress with deep learning models, the dynamic nature of real-world applications necessitates continual learning, where models adapt to new classes (plasticity) over time without forgetting old ones (catastrophic forgetting). Current continual segmentation methods often rely on distillation strategies like knowledge distillation and pseudo-labeling, which are effective but result in increased training complexity and computational overhead. In this paper, we introduce a novel and efficient method for continual panoptic segmentation based on Visual Prompt Tuning, dubbed ECLIPSE. Our approach involves freezing the base model parameters and fine-tuning only a small set of prompt embeddings, addressing both catastrophic forgetting and plasticity and significantly reducing the trainable parameters. To mitigate inherent challenges such as error propagation and semantic drift in continual segmentation, we propose logit manipulation to effectively leverage common knowledge across the classes. Experiments on ADE20K continual panoptic segmentation benchmark demonstrate the superiority of ECLIPSE, notably its robustness against catastrophic forgetting and its reasonable plasticity, achieving a new state-of-the-art.


## Updates    
**_2024-04-29_** First Commit, We release the official implementation of ECLIPSE.  


## Installation

Our implementation is based on [CoMFormer](https://github.com/fcdl94/CoMFormer) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).

Please check the [installation instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) and [dataset preparation](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).

You can see our core implementation from
- `mask2former/maskformer_model.py`
- `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py`

## Quick Start

1. Step t=0: Training the model for base classes (you can skip this process if you use pre-trained weights.)
2. Step t>1: Training the model for novel classes with ECLIPSE

|   Scenario   |  Script   |    Step-0 Weight    |  Final Weight |
| :----------------: | :----------------: | :------:  | :------:  |
| ADE20K-Panoptic 100-5  | `bash script/ade_ps/100_5.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_5_final.pth)|
| ADE20K-Panoptic 100-10  | `bash script/ade_ps/100_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_10_final.pth)|
| ADE20K-Panoptic 100-50  | `bash script/ade_ps/100_50.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_step0.pth) | [step1](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_50_final.pth)|
| | | | |
| ADE20K-Panoptic 50-10  | `bash script/ade_ps/50_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_10_final.pth)|
| ADE20K-Panoptic 50-20  | `bash script/ade_ps/50_20.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_20_final.pth)|
| ADE20K-Panoptic 50-50  | `bash script/ade_ps/50_50.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_step0.pth) | [step2](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_50_final.pth)|
| | | | |
| ADE20K-Semantic 100-5  | `bash script/ade_ss/100_5.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_5_final.pth)|
| ADE20K-Semantic 100-10  | `bash script/ade_ss/100_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_10_final.pth)|
| ADE20K-Semantic 100-50  | `reproduce error` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_step0.pth) | [step1](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_50_final.pth)|
| | | | |
| COCO-Panoptic 83-5  | `bash script/coco_ps/83_5.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_5_final.pth)|
| COCO-Panoptic 83-10  | `bash script/coco_ps/83_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_10_final.pth)|


<div align="center">

<img src="https://github.com/clovaai/ECLIPSE/releases/download/assets/adps.png" width="100%"/>
<br />
<br />

<img src="https://github.com/clovaai/ECLIPSE/releases/download/assets/cocops.png" width="100%"/>
<br />
<br />

<img src="https://github.com/clovaai/ECLIPSE/releases/download/assets/adss.png" width="100%"/>
</div>


## How to Cite
```
@InProceedings{Kim_2024_CVPR,
    author    = {Kim, Beomyoung and Yu, Joonsang and Hwang, Sung Ju},
    title     = {ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {3346-3356}
}
```

## License

```
ECLIPSE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
```
