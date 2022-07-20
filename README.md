## Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video [CVPR 2022]

Our **M**otion **P**ose and **S**hape **N**etwork (MPS-Net) is to effectively capture humans in motion to estimate accurate and temporally coherent 3D human pose and shape from a video.

Pleaser refer to our [arXiv report](https://arxiv.org/abs/2203.08534) for further details.

Check our YouTube videos below for 5 minute video presentation of our work.

[![PaperVideo](https://github.com/MPS-Net/MPS-Net_release/blob/main/Fig.png)](https://www.youtube.com/watch?v=lBZikM1vM60)

### Installation

This implementation has the demo and evaluation code for MPS-Net implemented in PyTorch.

MPS-Net has been implemented and tested on Ubuntu 18.04 with python >= 3.7. 

Clone the repo:
```bash
git clone https://github.com/MPS-Net/MPS-Net_release.git
```

Install the requirements using `virtualenv` or `conda`:
```bash
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

### Getting Started

First, you need download the required data (i.e our trained model and SMPL model parameters). To do this you can just run:

```bash
source scripts/prepare_data.sh
```

Download pre-processed data from [here](https://drive.google.com/drive/folders/1YTdq-9vP3E_eGDZXhxbHmxqDY6UIN_Cb?usp=sharing).

The data directory structure should follow the below hierarchy.
```
${ROOT}  
|-- data  
|   |-- base_data  
|   |-- preprocessed_data  
```

### Evaluation

Run the commands below to evaluate a pretrained model.
```bash
# dataset: 3dpw
python evaluate.py --dataset 3dpw --cfg ./configs/repr_table1_3dpw_model.yaml --gpu 0
```

Change the `TRAIN.PRETRAINED` field of the config file to the checkpoint you would like to evaluate.
You should be able to obtain the output below:

```shell script
# TRAIN.PRETRAINED = 'data/base_data/mpsnet_model_best.pth.tar'
...Evaluating on 3DPW test set...
PA-MPJPE: 52.1, MPJPE: 84.3, MPVPE: 99.7, ACC-ERR: 7.4
```

### Running the Demo

We have prepared a demo code to run MPS-Net on arbitrary videos. 
To do this you can just run:

```bash
# Run on a local video
python demo.py --vid_file sample_video.mp4 --gpu 0
```

### Citation

```bibtex
@inproceedings{WeiLin2022mpsnet,
  title={Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video},
  author={Wei, Wen-Li and Lin, Jen-Chun and Liu, Tyng-Luh and Liao, Hong-Yuan Mark},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2022}
}
```

### License
This project is licensed under the terms of the MIT license.

## References
The base codes are largely borrowed from great resources [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE).
