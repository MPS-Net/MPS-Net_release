# Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video

> [**Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video**](https://arxiv.org/abs/2203.08534),            
> [Wen-Li Wei](), [Jen-Chun Lin](https://sites.google.com/site/jenchunlin/), 
[Tyng-Luh Liu](https://homepage.iis.sinica.edu.tw/pages/liutyng/index_en.html), [Hong-Yuan Mark Liao](),        
> *IEEE Computer Vision and Pattern Recognition, 2022* 

## Getting Started

## Evaluation

- Download pre-trained MPS-Net weights from [here](https://drive.google.com/file/d/1GTy6uV5kgrhLv7Jpw8VDqDoeIVe9QC4Q/view?usp=sharing).  
```bash
# dataset: 3dpw
python evaluate.py --dataset 3dpw --cfg ./configs/table1_3dpw_model.yaml --gpu 0 
```

## License
This project is licensed under the terms of the MIT license.
