# Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video [CVPR 2022]

Our **M**otion **P**ose and **S**hape **N**etwork (MPS-Net) is to effectively capture humans in motion to estimate accurate and temporally coherent 3D human pose and shape from a video.

Pleaser refer to our [arXiv report](https://arxiv.org/abs/2203.08534) for further details.

Check our YouTube video below for 5 minute video presentation of our work.

[![PaperVideo](https://github.com/MPS-Net/MPS-Net/blob/gh-pages/Fig.png)](https://www.youtube.com/watch?v=lBZikM1vM60)

## Getting Started 

#### Installation & Clone the repo [Environment on Linux (Ubuntu 18.04 with python >= 3.7)]

```bash
# Clone the repo:
git clone https://github.com/MPS-Net/MPS-Net_release.git

# Install the requirements using `virtualenv`: 
cd $PWD/MPS-Net_release
source scripts/install_pip.sh
```
or

#### Installation & Clone the repo [Windows + Anaconda + Git Bash]

```bash
# Download and installing Anaconda on Windows:  https://www.anaconda.com/products/distribution#windows

# Installing Git Bash: 
cmd
winget install --id Git.Git -e --source winget

# Launch Git Bash
start "" "%PROGRAMFILES%\Git\bin\sh.exe" --login

# Clone the repo:
git clone https://github.com/MPS-Net/MPS-Net_release.git

# Install the requirements using `conda`: 
cd MPS-Net_release
source scripts/install_conda.sh
```

## Download the Required Data 

You can just run:

```bash
source scripts/get_base_data.sh
```

or

You can download the required data and the pre-trained MPS-Net model from [here](https://drive.google.com/drive/folders/1YTdq-9vP3E_eGDZXhxbHmxqDY6UIN_Cb?usp=sharing). 
You need to unzip the contents and the data directory structure should follow the below hierarchy.

```
${ROOT}  
|-- data  
|   |-- base_data  
|   |-- preprocessed_data  
```

## Evaluation

Run the commands below to evaluate a pretrained model on 3DPW test set.

```bash
# dataset: 3dpw
python evaluate.py --dataset 3dpw --cfg ./configs/repr_table1_3dpw_model.yaml --gpu 0
```

You should be able to obtain the output below:

```shell script
PA-MPJPE: 52.1, MPJPE: 84.3, MPVPE: 99.7, ACC-ERR: 7.4
```

## Running the Demo

We have prepared a demo code to run MPS-Net on arbitrary videos. 
To do this you can just run:

```bash
python demo.py --vid_file sample_video.mp4 --gpu 0
```

sample_video.mp4 demo output:

<p float="left">
  <img src="assets/MPS-Net_sample_video_output.gif" width="30%" />
</p>

```bash
python demo.py --vid_file sample_video2.mp4 --gpu 0
```

sample_video2.mp4 demo output:

<p float="left">
  <img src="assets/MPS-Net_sample_video2_output.gif" width="30%" />
</p>

## Citation

```bibtex
@inproceedings{WeiLin2022mpsnet,
  title={Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video},
  author={Wei, Wen-Li and Lin, Jen-Chun and Liu, Tyng-Luh and Liao, Hong-Yuan Mark},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2022}
}
```

## License
This project is licensed under the terms of the MIT license.

## References
The base codes are largely borrowed from great resources [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE).
