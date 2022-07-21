#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=11LXqYqjLKBn2wOcNbDT7DX0Hbj80ynbp&export=download&confirm=t"
unzip base_data.zip
rm base_data.zip

gdown "https://drive.google.com/uc?id=1d6NV2eBj8FVTs-7c-MUGM-YOGuT5pZH_&export=download&confirm=t"
unzip preprocessed_data.zip
rm preprocessed_data.zip

cd ..
mkdir -p $HOME/.torch/models/
mv data/base_data/yolov3.weights $HOME/.torch/models/
