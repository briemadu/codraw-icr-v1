#!/bin/sh

python3 main.py -n_epochs 200

python3 main.py
python3 main.py -no_context
python3 main.py -no_image
python3 main.py -no_msg
python3 main.py -text_pretrained all-mpnet-base-v2_no-teller
python3 main.py -text_pretrained all-mpnet-base-v2_no-drawer

python3 main.py -task teller
python3 main.py -no_context -task teller
python3 main.py -no_image -task teller
python3 main.py -no_msg -task teller
python3 main.py -text_pretrained all-mpnet-base-v2_no-teller -task teller
python3 main.py -text_pretrained all-mpnet-base-v2_no-drawer -task teller