#!/usr/bin/env bash

sudo apt install python3.7-venv
echo "Creating virtual environment"
python3.7 -m venv mps-env
echo "Activating virtual environment"

source $PWD/mps-env/bin/activate

python -m pip install --upgrade pip

$PWD/mps-env/bin/pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
$PWD/mps-env/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/mps-env/bin/pip install -r requirements.txt
