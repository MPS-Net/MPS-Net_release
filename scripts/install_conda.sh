#!/usr/bin/env bash

export CONDA_ENV_NAME=mps-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

python -m pip install --upgrade pip setuptools wheel

which python
which pip

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install numpy==1.17.5
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r requirements.txt
pip install gdown
