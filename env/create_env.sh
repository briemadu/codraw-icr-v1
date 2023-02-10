#!/bin/bash

conda create -n codraw_pl python=3.9
conda activate codraw_pl
conda install pytorch torchvision torchtext -c pytorch
pip install pytorch-lightning
conda install h5py
pip install comet-ml
pip install torchmetrics
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install sentence-transformers
conda install ipykernel
pip install torchmetrics --upgrade