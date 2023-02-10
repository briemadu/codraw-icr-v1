#!/bin/bash

sh pathprep.sh

nano codrawmodels/__init__.py

mkdir checkpoints
mkdir outputs

python3 data/preprocessed/img_embeddings.py
python3 data/preprocessed/text_embeddings.py
python3 data/preprocessed/text_embeddings.py -remove_teller
python3 data/preprocessed/text_embeddings.py -remove_drawer
