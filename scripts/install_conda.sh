#!/usr/bin/env bash

conda create -n mps-env python=3.7

conda activate mps-env

pip install numpy==1.17.5 torch==1.4.0 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.5.0
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r requirements.txt
