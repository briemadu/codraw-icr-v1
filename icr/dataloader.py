#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load all the necessary CoDraw data, iCR labels and pretrained embeddings into
a Pytorch Dataset object.

One class to handle the images, one to handle the text embeddings and one to
store the complete dataset.
"""

import argparse
import random
from collections import namedtuple
from pathlib import Path
from typing import Dict, Set, Tuple

import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from icr.config import LABEL_MAP
from icr.structs.game import Game


Datapoint: Tuple[Tensor, Tensor, Tensor, int] = namedtuple(
    'Datapoint', ['context', 'last_msg', 'img', 'label'])
Constructed = Tuple[Dict[int, Game], Dict[int, Tuple[int, int]]]
LoaderItem = Tuple[int, Tensor, Tensor, Tensor, int]


class Images():
    """Contain the pretrained image embeddings."""
    def __init__(self, preproc_path: str, pretrained: str,
                 split: str, task: str, delay_scenes: bool):
        self.pretrained = pretrained
        self.split = split
        self.task = task
        self.delay_scenes = delay_scenes
        self.data = self.load(preproc_path, pretrained)

    def load(self, preproc_path: str, pretrained: str) -> Dict[int, np.array]:
        """Read and store image embeddings."""
        directory = Path(f'{preproc_path}{pretrained}/')
        which = '' if self.task == 'drawer' else 'orig-'
        fname = directory / f'codraw_{which}img-embeddings_{self.split}.hdf5'
        file = h5py.File(fname, 'r')
        embeddings = {int(key): value[:] for key, value in file.items()}
        file.close()
        if self.task == 'drawer' and self.delay_scenes:
            embeddings = self._add_delay(embeddings)
        return embeddings

    def _add_delay(self, embs: Dict[int, np.array]) -> Dict[int, np.array]:
        """Add an empty scene at the beginning and shift the whole sequence."""
        # since we have no fine-grained annotation about when the drawer's
        # message was written, we'd better assume the drawer first do all the
        # actions s/he can before sending the iCR
        # if we consider that the iCR is formulated before any action is taken,
        # (which is unlikely) we thus need to shift the images so that at step
        # 0 the image the drawer sees is always empty and then it must see the
        # image state at the beginning of the current round
        # in this case, so we add an empty scene to all sequences

        # FIXME: this is a hack to get an empty scene embedding
        # by manually retrieving one from dialogues that contain an empty
        # first scene at the beginning
        if self.split == 'train':
            index = 0
        if self.split == 'val':
            index = 8
        if self.split == 'test':
            index = 9
        empty_scene = np.expand_dims(embs[index][0], axis=0)
        shifted_embs = {key: np.concatenate([empty_scene, emb])
                        for key, emb in embs.items()}
        return shifted_embs

    def get_image(self, game_id: int, turn: int) -> Tensor:
        """Retrieve an image representation."""
        if self.task == 'drawer':
            img = torch.tensor(self.data[game_id][turn])
        elif self.task == 'teller':
            img = torch.tensor(self.data[game_id])
        img = img.float().squeeze(0)
        return img


class TextEmbeddings():
    """Contain context and utterance embeddings."""
    def __init__(self, preproc_path: str, pretrained: str, split: str,
                 task: str):
        self.pretrained = pretrained
        self.split = split
        self.task = task
        self.contexts = self.load_contexts(preproc_path, pretrained)
        self.last_msgs = self.load_msgs(preproc_path, pretrained)

    def load_contexts(self, path: str, pretrained: str) -> Dict[int, np.array]:
        """Read and store the cumulative context embeddings."""
        directory = Path(f'{path}{pretrained}/')
        fname = directory / f'codraw_dialogues_{self.split}.hdf5'
        file = h5py.File(fname, 'r')
        cumul_embs = {int(key): value[:] for key, value in file.items()}
        file.close()
        return cumul_embs

    def load_msgs(self, path: str, pretrained: str) -> Dict[int, np.array]:
        """Read and store utterance embeddings."""
        directory = Path(f'{path}{pretrained}/')
        fname = directory / f'codraw_utterances_{self.split}.hdf5'
        file = h5py.File(fname, 'r')
        utt_embs = {int(key): value[:] for key, value in file.items()}
        file.close()
        return utt_embs

    def get_last_msg(self, game_id: int, turn: int) -> Tensor:
        """Retrive an utterance's representation."""
        if self.task == 'drawer':
            # get the last teller's utterance that the drawer has to classify
            dim = 0
        elif self.task == 'teller':
            # get the last drawer's utterance that the teller has to classify
            dim = 1
        return torch.tensor(self.last_msgs[game_id][turn][dim])

    def get_context(self, game_id: int, turn: int) -> Tensor:
        """Retrieve a context's representation."""
        if self.task == 'drawer':
            # get context up to the teller's utterance
            dim = 0
        elif self.task == 'teller':
            # get context after teller's and up to the drawer's utterance
            dim = 1
        return torch.tensor(self.contexts[game_id][turn][dim])


class CodrawData(Dataset):
    """Build all CoDraw datapoints."""
    def __init__(self, split: str, config: argparse.Namespace,
                 quick_load: bool = True):
        self.quick_load = quick_load
        self.split = split
        self.labels_dic = LABEL_MAP
        self.extract_config(config)
        self.embs = TextEmbeddings(config.path_to_preprocessed_texts,
                                   config.text_pretrained,
                                   split, config.task)
        self.images = Images(config.path_to_preprocessed_imgs,
                             config.img_pretrained,
                             split, config.task, config.delay_scenes)
        self.games, self.datapoints = self._construct(
            config.path_to_codraw, config.path_to_annotation,
            config.annotation_column_name)
        self.sizes = self.stats()

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> LoaderItem:
        game_id, turn = self.datapoints[idx]
        datap = self._get_datapoint(game_id, turn)
        return idx, datap.context, datap.last_msg, datap.img, datap.label

    @property
    def n_labels(self) -> int:
        """Return number of class labels."""
        return len(self.labels_dic)

    def extract_config(self, config: argparse.Namespace) -> None:
        """Extract necessary hyperparameters."""
        self.task = config.task
        self.downsample = config.downsample
        self.upsample = config.upsample
        self.only_with_icrs = config.only_with_icrs
        self.include_peek = not config.only_until_peek
        self.remove_first = config.remove_first

    def _construct(self, path_to_codraw: str, path_to_annotation: str,
                   column_name: str) -> Constructed:
        """Load all information and create datapoints dictionary."""
        codraw = self._load_codraw(path_to_codraw)
        icrs = self._load_icrs(path_to_annotation, column_name)
        datapoints: Dict[int, Tuple[int, int]] = {}
        games = {}
        for name, game_data in codraw['data'].items():
            if self.split not in name:
                continue
            game_idx = int(name.split('_')[1])
            game = Game(name, game_data, icrs, quick_load=self.quick_load,
                        include_peek=self.include_peek)
            if self.only_with_icrs and not game.dialogue.icr_turns:
                # ignore dialogues without iCRs
                continue
            games[game_idx] = game
            for turn in range(game.n_turns):
                if turn == 0 and self.remove_first:
                    continue
                label = self.get_label(game, turn)
                # downsample negative label on train set
                if (self.split == 'train'
                        and label == self.labels_dic['not_icr']
                        and random.random() > self.downsample):
                    # skip the datapoint
                    continue
                idx = len(datapoints)
                datapoints[idx] = (game_idx, turn)
                if (self.split == 'train'
                        and label == self.labels_dic['icr']
                        and self.upsample > 0):
                    # duplicate the datapoint
                    for _ in range(self.upsample):
                        idx = len(datapoints)
                        datapoints[idx] = (game_idx, turn)
        return games, datapoints

    def stats(self) -> Dict[str, float]:
        """Print and store basic descriptive statistics."""
        n_icr = sum([1 for idx in range(len(self))
                    if self[idx][-1] == self.labels_dic['icr']])
        n_other = sum([1 for idx in range(len(self))
                       if self[idx][-1] == self.labels_dic['not_icr']])
        perc_icr = 100 * n_icr / len(self)
        perc_other = 100 * n_other / len(self)

        print(f'{"---"*10}\n Loaded {self.split} set with:')
        print(f'   {len(self)} datapoints')
        print(f'   {len(self.games)} dialogues')
        print(f'   {n_icr} ({perc_icr:.2f}%) clarifications')
        print(f'   {n_other} ({perc_other:.2f}%) other \n{"---"*10}')

        stats = {f'{self.split}: n_icr': n_icr,
                 f'{self.split}: %_icr': perc_icr,
                 f'{self.split}: n_other': n_other,
                 f'{self.split}: %_other': perc_other,
                 f'{self.split}: n_games': len(self.games),
                 f'{self.split}: n_datapoints': len(self)}
        return stats

    @staticmethod
    def _load_codraw(path_to_codraw: str) -> dict:
        """Read CoDraw JSON file."""
        with open(path_to_codraw, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def _load_icrs(path_to_annotation: str, column: str) -> Set[str]:
        """Read CR labels."""
        annotated = pd.read_csv(path_to_annotation, sep='\t')
        sentences = annotated['drawer\'s utterance']
        labels = annotated[column]
        icrs = {sent for sent, label in zip(sentences, labels)
                if label == LABEL_MAP['icr']}
        return icrs

    def _get_datapoint(self, game_id: int, turn: int) -> Datapoint:
        """Retrieve a datapoint's four components."""
        game = self.games[game_id]
        context = self.embs.get_context(game_id, turn)
        last_msg = self.embs.get_last_msg(game_id, turn)
        img = self.images.get_image(game_id, turn)
        label = self.get_label(game, turn)
        return Datapoint(context=context, last_msg=last_msg,
                         img=img, label=label)

    def get_label(self, game: Game, turn: int) -> int:
        """Retrieve a label."""
        return (self.labels_dic['icr'] if turn in game.dialogue.icr_turns
                else self.labels_dic['not_icr'])
