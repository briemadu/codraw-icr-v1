#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to turn CoDraw png scenes (original and the stepwise reconstructions)
into h5py objects with numpy array representations of the images.
Also create a h5py object with the pretrained image embeddings.

The constants MODEL_NAME and SHAPE can be passed via the CLI. This script
is meant to be run from the root of this repository.

Based on tutorial: https://pytorch.org/vision/0.12/models.html
and this forum answer:
https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198

For each split, this script builds 4 h5py objects. All of them contain one
dataset for each scene ID in CoDraw.
1. codraw_images_{SPLIT}.hdf5
2. codraw_img-embeddings_{SPLIT}.hdf5
3. codraw_orig-images_{SPLIT}.hdf5
4. codraw_orig-img-embeddings_{SPLIT}.hdf5

* For the stepwise reconstructions:
In 1, each dataset contains a tensor of dim (n_rounds, 3, 400, 500),
where 3 is the number of RGB channels, 400 and 500 are the image dimensions.
In 2, each dataset contains a tensor of dim (n_rounds, 1, SHAPE),
where SHAPE is the model's embedding dimension.
In both, the index of n_rounds corresponds to the game round.

* For the originals:
In 3, each dataset contains a tensor of dim (3, 400, 500), where
3 is the number of RGB channels, 400 and 500 and the image dimensions.
In 4, each dataset contains a tensor of dim (1, SHAPE),
where SHAPE is the model's embedding dimension.
"""

import argparse
import os
from pathlib import Path
import h5py
import json
import numpy as np
from tqdm import tqdm
import torch
from torchvision.io import read_image
from torchvision import transforms
import torchvision
from torchvision.models import (resnet18, vgg16, googlenet, inception_v3,
                                alexnet, resnet101, vgg19_bn)

# Read model, dimension and whether to resize and center images
argParser = argparse.ArgumentParser()
argParser.add_argument("-model", type=str, default='resnet101',
                       help="Pretrained model name")
argParser.add_argument("-dimension", type=int, default=2048,
                       help="Pretrained model output dim")
argParser.add_argument("-resize_and_center", action='store_true',
                       help="Preprocess image as in documentation.")
args = argParser.parse_args()

MODEL_NAME = args.model
SHAPE = args.dimension
RESIZE_AND_CENTER = args.resize_and_center
print(f'Generating image embeddings using {MODEL_NAME}.')
print(f'Resize and center is set to {RESIZE_AND_CENTER}.')

ORIGIN = './data/'
CODRAW_PATH = Path(f'{ORIGIN}CoDraw-master/dataset/CoDraw_1_0.json')
# sequential reconstructions by the drawer
IMAGES = Path(f'{ORIGIN}IncrementalCoDrawImages/DrawerScenes/')
# original image seen by the teller
ORIG_IMAGES = Path(f'{ORIGIN}AbstractScenes_v1.1/RenderedScenes/')
SPLITS = ('train', 'val', 'test')
DEVICE = 'cuda'

DEST = f'{ORIGIN}preprocessed/images/'
try:
    os.mkdir(DEST)
except FileExistsError:
    pass
adj = '_resized_centered' if RESIZE_AND_CENTER else ''
OUTPUT_DIR = Path(f'{DEST}{MODEL_NAME}{adj}/')
os.mkdir(OUTPUT_DIR)
OUTPUT_RAW = Path(f'{DEST}raw/')
try:
    os.mkdir(OUTPUT_RAW)
except FileExistsError:
    print('Raw file will be overwritten!')


# store numpy representations
INC_MATRICES = 'codraw_images_{}.hdf5'
ORIG_MATRICES = 'codraw_orig-images_{}.hdf5'
# store embedding vectors
INC_EMBS = 'codraw_img-embeddings_{}.hdf5'
ORIG_EMBS = 'codraw_orig-img-embeddings_{}.hdf5'

models = {'resnet18': resnet18, 'vgg16': vgg16, 'googlenet': googlenet,
          'inception_v3': inception_v3, 'alexnet': alexnet,
          'resnet101': resnet101, 'vgg19_bn': vgg19_bn}

image_encoder = models[MODEL_NAME](pretrained=True).to(DEVICE)
# We don't need the last classification layer, only the features
# Following PS' example code and https://discuss.pytorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-my-own-dataset/9008/6
modules = list(image_encoder.children())[:-1]
image_encoder = torch.nn.Sequential(*modules)
image_encoder.eval()

mode = torchvision.io.image.ImageReadMode.RGB
# following the documentation link above, we convert the uint8 RGB tensor
# to a float tensor between 0 and 1 and then normalize it
convert = transforms.ConvertImageDtype(torch.float)
resize = transforms.Resize(256)
center = transforms.CenterCrop(224)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
if RESIZE_AND_CENTER:
    preprocess = transforms.Compose([convert, resize, center, normalize])
else:
    preprocess = transforms.Compose([convert, normalize])

with open(CODRAW_PATH, 'r') as file:
    codraw = json.load(file)

# collect all game IDs and number of rounds on each split
idxs = {split: {} for split in SPLITS}
for name, game in codraw['data'].items():
    split, game_id = name.split('_')
    game_id = str(int(game_id))
    n_rounds = len(game['dialog'])
    idxs[split][game_id] = n_rounds
# collect all scenes and turns in the directory
img_seqs = {}
for name in os.listdir(IMAGES):
    if name.endswith('.png'):
        scene_id, round_ = name.strip('Scene').strip('.png').split('_')
        if scene_id not in img_seqs:
            img_seqs[scene_id] = []
        img_seqs[scene_id].append(round_)
# make sure that scenes and rounds in codraw and the reconstructed scenes match
for split in SPLITS:
    for idx, n in idxs[split].items():
        # same number of turns
        assert len(img_seqs[idx]) == n
        assert len(set(img_seqs[idx])) == len(img_seqs[idx])


def preprocess_and_forward(image, model):
    """Preprocess image and extract representation from the model.

    Args:
        img (torch.Tensor): A tensor representing the image png.

    Returns:
        np.array: image embedding vector.
    """
    # confirm that the conversion from uint8 to float makes values be in
    # the interval [0,1], as the documentation requires
    assert torch.min(convert(image)) >= 0
    assert torch.max(convert(image)) <= 1
    # preprocess and add a dummy dimension for the batch
    x = preprocess(image).to(DEVICE).unsqueeze(0)
    encoded_image = model(x)
    # remove dummy dimensions
    encoded_image = encoded_image.squeeze(-1).squeeze(-1)
    return encoded_image.cpu().numpy()


# Extract image arrays and embeddings for the incremental scenes (drawer)
with torch.no_grad():
    for split in SPLITS:
        output_images = OUTPUT_RAW / INC_MATRICES.format(split)
        output_embeddings = OUTPUT_DIR / INC_EMBS.format(split)
        with h5py.File(output_images, 'w') as out_img:
            with h5py.File(output_embeddings, 'w') as out_emb:
                for game_id, n_rounds in tqdm(idxs[split].items()):
                    sequence = []  # store the arrays
                    sequence_embeddings = []  # store the embedding vectors
                    # loop over round indexes to make sure the order is correct
                    for turn in range(n_rounds):
                        img_name = f'Scene{game_id}_{turn}.png'
                        # img has shape torch.Size([3, 400, 500])
                        img = read_image(str(IMAGES / img_name), mode=mode)
                        sequence.append(img.numpy())
                        output = preprocess_and_forward(img, image_encoder)
                        sequence_embeddings.append(output)
                    sequence = np.array(sequence)
                    sequence_embeddings = np.array(sequence_embeddings)
                    assert sequence.shape == (n_rounds, 3, 400, 500)
                    assert sequence_embeddings.shape == (n_rounds, 1, SHAPE)
                    out_img.create_dataset(game_id, data=sequence)
                    out_emb.create_dataset(game_id, data=sequence_embeddings)

# Extract image arrays and embeddings for the original scenes (teller)
counter = 0
with torch.no_grad():
    for split in SPLITS:
        output_images = OUTPUT_RAW / ORIG_MATRICES.format(split)
        output_embeddings = OUTPUT_DIR / ORIG_EMBS.format(split)
        with h5py.File(output_images, 'w') as out_img:
            with h5py.File(output_embeddings, 'w') as out_emb:
                for game_id, _ in tqdm(idxs[split].items()):
                    # scene names have a _, we need to hack a 0 into first 10
                    part1, part2 = game_id[:-1], game_id[-1]
                    if part1 == '':
                        counter += 1
                        part1 = 0
                    img_name = f'Scene{part1}_{part2}.png'
                    # img has shape torch.Size([3, 400, 500])
                    img = read_image(str(ORIG_IMAGES / img_name), mode=mode)
                    output = preprocess_and_forward(img, image_encoder)
                    img = img.numpy()
                    assert img.shape == (3, 400, 500)
                    assert output.shape == (1, SHAPE)
                    out_img.create_dataset(game_id, data=img)
                    out_emb.create_dataset(game_id, data=output)
    assert counter == 10
