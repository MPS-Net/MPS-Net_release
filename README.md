## Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video

This repository is the official [Pytorch](https://pytorch.org/) implementation of MPS-Net. 
For more results and video demos can be found at [here](https://mps-net.github.io/MPS-Net/).
The base codes are largely borrowed from [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE).

> [**Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video**](https://arxiv.org/abs/2203.08534),            
> [Wen-Li Wei*](), [Jen-Chun Lin*](https://sites.google.com/site/jenchunlin/), 
[Tyng-Luh Liu](https://homepage.iis.sinica.edu.tw/pages/liutyng/index_en.html), [Hong-Yuan Mark Liao](),        
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2022* 

> *Both authors contributed equally to this work

### Getting Started


### Evaluation

First, you need download the required data(i.e our trained model and SMPL model parameters). To do this you can just run:

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

Run the commands below to evaluate a pretrained model.
```bash
# dataset: 3dpw
python evaluate.py --dataset 3dpw --cfg ./configs/repr_table1_3dpw_model.yaml --gpu 0 
```
- Download pre-trained MPS-Net weights from [here](https://drive.google.com/file/d/1GTy6uV5kgrhLv7Jpw8VDqDoeIVe9QC4Q/view?usp=sharing).  

You should be able to obtain the output below:

```shell script
...Evaluating on 3DPW test set...
PA-MPJPE: 52.1, MPJPE: 84.3, MPVPE: 99.7, ACC-ERR: 7.4
```

### Running the Demo

### License
This project is licensed under the terms of the MIT license.

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
