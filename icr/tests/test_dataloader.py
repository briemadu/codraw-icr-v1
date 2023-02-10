#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/dataloader.py.
"""

import unittest
from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torchvision.io import read_image
from torchvision import transforms
import torchvision
from torchvision.models import resnet101

from icr.dataloader import CodrawData, Images, TextEmbeddings

CODRAW_PATH = './data/CoDraw-master/dataset/CoDraw_1_0.json'
PRETRAINED = 'all-mpnet-base-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_SEQ_LEN = 200
SEP_TELLER = '/T '
SEP_DRAWER = '/D '
SEP_PEEK = '/PEEK'

PRETRAINED_IMG = 'resnet101'
IMAGES = Path('./data/IncrementalCoDrawImages/DrawerScenes/')
ORIG_IMAGES = Path('./data/AbstractScenes_v1.1/RenderedScenes/')


class TestTextEmbeddings(unittest.TestCase):
    """Test the CoDraw text embedding container."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)['data']
        # load Sentence Transformer, same checkpoint
        cls.encoder = SentenceTransformer(PRETRAINED).to(DEVICE)

    def setUp(self):
        self.drawer_embs = TextEmbeddings(
            preproc_path='./data/preprocessed/texts/',
            pretrained=PRETRAINED,
            split='val',
            task='drawer')

        self.teller_embs = TextEmbeddings(
            preproc_path='./data/preprocessed/texts/',
            pretrained=PRETRAINED,
            split='train',
            task='teller')

    def test_get_last_msg(self):
        # this test may fail in case a message is longer than the maximum
        # number of input tokens we defined for encoding, but that
        # should not matter as messages can probably not be longer than 200
        # tokens
        for game_idx in ('val_07028', 'val_06978', 'val_07608', 'val_09238'):
            n = len(self.codraw[game_idx]['dialog'])
            idx = int(game_idx.split('_')[1])
            for turn in range(n):
                utterance = self.codraw[game_idx]['dialog'][turn]['msg_t']
                embedding = torch.tensor(self.encoder.encode([utterance]))
                loaded = self.drawer_embs.get_last_msg(idx, turn)
                self.assertTrue(torch.equal(embedding.squeeze(0), loaded))

        for game_idx in ('train_09231', 'train_09195', 'train_08142',
                         'train_00005'):
            n = len(self.codraw[game_idx]['dialog'])
            idx = int(game_idx.split('_')[1])
            for turn in range(n):
                utterance = self.codraw[game_idx]['dialog'][turn]['msg_d']
                embedding = torch.tensor(self.encoder.encode([utterance]))
                loaded = self.teller_embs.get_last_msg(idx, turn)
                self.assertTrue(torch.equal(embedding.squeeze(0), loaded))

    def _merge_context(self, dialogue, n_turn, task):
        """Auxiliary function to create the context."""
        context = ''
        for t in range(n_turn + 1):
            turn = dialogue[t]
            if t == n_turn and task == 'drawer':
                return self._truncate(context)
            if 'peeked' in turn:
                context += f' {SEP_PEEK}'
            context += f' {SEP_TELLER}{turn["msg_t"]}'
            if t == n_turn and task == 'teller':
                return self._truncate(context)
            context += f' {SEP_DRAWER}{turn["msg_d"]}'

    @staticmethod
    def _truncate(text):
        """Truncate left context, otherwise model probably truncates right."""
        return " ".join(text.split()[-MAX_SEQ_LEN:])

    def test_get_context(self):
        for game_idx in ('val_07028', 'val_06978', 'val_07608', 'val_09238'):
            n = len(self.codraw[game_idx]['dialog'])
            idx = int(game_idx.split('_')[1])
            for turn in range(n):
                context = self._merge_context(self.codraw[game_idx]['dialog'],
                                              turn, 'drawer')
                embedding = torch.tensor(self.encoder.encode([context]))
                loaded = self.drawer_embs.get_context(idx, turn)
                self.assertTrue(torch.equal(embedding.squeeze(0), loaded))

        for game_idx in ('train_09231', 'train_09195', 'train_08142',
                         'train_00005'):
            n = len(self.codraw[game_idx]['dialog'])
            idx = int(game_idx.split('_')[1])
            for turn in range(n):
                context = self._merge_context(self.codraw[game_idx]['dialog'],
                                              turn, 'teller')
                embedding = torch.tensor(self.encoder.encode([context]))
                loaded = self.teller_embs.get_context(idx, turn)
                self.assertTrue(torch.equal(embedding.squeeze(0), loaded))


class TestImages(unittest.TestCase):
    """Test the CoDraw image embedding container."""

    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)['data']
        # load Sentence Transformer, same checkpoint
        encoder = resnet101(pretrained=True).to(DEVICE).eval()
        modules = list(encoder.children())[:-1]
        cls.encoder = torch.nn.Sequential(*modules).eval()

        cls.mode = torchvision.io.image.ImageReadMode.RGB
        convert = transforms.ConvertImageDtype(torch.float)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        cls.preprocess = transforms.Compose([convert, normalize])

    def setUp(self):
        self.drawer_embs = Images(
            preproc_path='./data/preprocessed/images/',
            pretrained=PRETRAINED_IMG,
            split='val',
            task='drawer',
            delay_scenes=False)

        self.delayed_drawer_embs = Images(
            preproc_path='./data/preprocessed/images/',
            pretrained=PRETRAINED_IMG,
            split='val',
            task='drawer',
            delay_scenes=True)

        self.teller_embs = Images(
            preproc_path='./data/preprocessed/images/',
            pretrained=PRETRAINED_IMG,
            split='val',
            task='teller',
            delay_scenes=False)

    def test_get_image(self):
        for game_idx in ('val_07028', 'val_06978', 'val_07608', 'val_09238'):
            n = len(self.codraw[game_idx]['dialog'])
            idx = int(game_idx.split('_')[1])

            part1, part2 = str(idx)[:-1], str(idx)[-1]
            img_name = f'Scene{part1}_{part2}.png'
            img = read_image(str(ORIG_IMAGES / img_name), mode=self.mode)
            x = self.preprocess(img).to(DEVICE).unsqueeze(0)
            encoded_image = self.encoder(x)
            # remove dummy dimensions
            teller_embedding = encoded_image.squeeze(-1).squeeze(-1).squeeze(0)

            for turn in range(n):
                img_name = f'Scene{idx}_{turn}.png'
                img = read_image(str(IMAGES / img_name), mode=self.mode)
                x = self.preprocess(img).to(DEVICE).unsqueeze(0)
                encoded_image = self.encoder(x)
                # remove dummy dimensions
                embedding = encoded_image.squeeze(-1).squeeze(-1).squeeze(0)
                loaded = self.drawer_embs.get_image(idx, turn)
                self.assertTrue(torch.equal(embedding.cpu(), loaded))

                loaded = self.teller_embs.get_image(idx, turn)
                self.assertTrue(torch.equal(teller_embedding.cpu(), loaded))

    def test_add_delay(self):
        # get one empty scene for comparison
        empty = self.drawer_embs.data[7028][0]
        for key, embs in self.drawer_embs.data.items():
            delayed_portion = self.delayed_drawer_embs.data[key][1:]
            np.testing.assert_array_equal(embs, delayed_portion)
            np.testing.assert_array_equal(
                empty, self.delayed_drawer_embs.data[key][0])


@dataclass
class Parameters:
    """Auxiliar dataclass to replace the config.args(), which throws an error
    when combined with the unittest arguments in the CLI."""

    task: str = 'drawer'
    downsample: float = 1.0
    upsample: int = 0
    only_with_icrs: bool = False
    only_until_peek: bool = False
    remove_first: bool = False
    no_msg: bool = False
    delay_scenes: bool = False

    path_to_codraw: str = './data/CoDraw-master/dataset/CoDraw_1_0.json'
    path_to_annotation: str = './data/cr_anno_adjudication/data_adju.tsv'
    path_to_preprocessed_imgs: str = './data/preprocessed/images/'
    path_to_preprocessed_texts: str = './data/preprocessed/texts/'
    annotation_column_name: str = 'is CR? v2'

    text_pretrained: str = 'all-mpnet-base-v2'
    img_pretrained: str = 'resnet101'


class TestCodrawData(unittest.TestCase):
    """Test the CoDraw dataset container."""
    @classmethod
    def setUpClass(cls):
        cls.params = Parameters()
        cls.train_d = CodrawData('train', cls.params)
        cls.val_d = CodrawData('val', cls.params)
        cls.test_d = CodrawData('test', cls.params)

        cls.params_t = Parameters()
        cls.params_t.task = 'teller'
        cls.train_t = CodrawData('train', cls.params_t)
        cls.val_t = CodrawData('val', cls.params_t)
        cls.test_t = CodrawData('test', cls.params_t)

    def test_build(self):
        self.assertEqual(len(self.train_d), 62067)
        self.assertEqual(len(self.val_d), 7714)
        self.assertEqual(len(self.test_d), 7721)
        self.assertEqual(len(self.train_t), 62067)
        self.assertEqual(len(self.val_t), 7714)
        self.assertEqual(len(self.test_t), 7721)
        # correct number of games
        self.assertEqual(len(self.train_d.games), 7989)
        self.assertEqual(len(self.val_d.games), 1002)
        self.assertEqual(len(self.test_d.games), 1002)
        self.assertEqual(len(self.train_t.games), 7989)
        self.assertEqual(len(self.val_t.games), 1002)
        self.assertEqual(len(self.test_t.games), 1002)
        # as many image sets as games
        self.assertEqual(len(self.train_d.images.data), 7989)
        self.assertEqual(len(self.val_d.images.data), 1002)
        self.assertEqual(len(self.test_d.images.data), 1002)
        self.assertEqual(len(self.train_t.images.data), 7989)
        self.assertEqual(len(self.val_t.images.data), 1002)
        self.assertEqual(len(self.test_t.images.data), 1002)
        # as many text embedding sets as games
        self.assertEqual(len(self.train_d.embs.contexts), 7989)
        self.assertEqual(len(self.val_d.embs.contexts), 1002)
        self.assertEqual(len(self.test_d.embs.contexts), 1002)
        self.assertEqual(len(self.train_d.embs.last_msgs), 7989)
        self.assertEqual(len(self.val_d.embs.last_msgs), 1002)
        self.assertEqual(len(self.test_d.embs.last_msgs), 1002)

        self.assertEqual(len(self.train_t.embs.contexts), 7989)
        self.assertEqual(len(self.val_t.embs.contexts), 1002)
        self.assertEqual(len(self.test_t.embs.contexts), 1002)
        self.assertEqual(len(self.train_t.embs.last_msgs), 7989)
        self.assertEqual(len(self.val_t.embs.last_msgs), 1002)
        self.assertEqual(len(self.test_t.embs.last_msgs), 1002)
        # splits not mixed
        set1 = set(self.train_d.images.data.keys())
        set2 = set(self.val_d.images.data.keys())
        set3 = set(self.test_d.images.data.keys())
        self.assertFalse(set1.intersection(set2))
        self.assertFalse(set1.intersection(set3))
        self.assertFalse(set2.intersection(set3))
        set1 = set(self.train_t.images.data.keys())
        set2 = set(self.val_t.images.data.keys())
        set3 = set(self.test_t.images.data.keys())
        self.assertFalse(set1.intersection(set2))
        self.assertFalse(set1.intersection(set3))
        self.assertFalse(set2.intersection(set3))
        # datapoint indexes
        for i in range(0, len(self.train_d)):
            self.assertTrue(i in self.train_d.datapoints)
            self.assertTrue(i in self.train_t.datapoints)
        for i in range(0, len(self.val_d)):
            self.assertTrue(i in self.val_d.datapoints)
            self.assertTrue(i in self.val_t.datapoints)
        for i in range(0, len(self.test_d)):
            self.assertTrue(i in self.test_d.datapoints)
            self.assertTrue(i in self.test_t.datapoints)

        # alternate t and d because total is tested above
        n_icrs = len([1 for i in range(len(self.train_d))
                      if self.train_d[i][-1] == 1])
        n_other = len([1 for i in range(len(self.train_t))
                       if self.train_t[i][-1] == 0])
        self.assertEqual(n_icrs, 7016)
        self.assertEqual(n_other, 55051)

        n_icrs = len([1 for i in range(len(self.val_d))
                      if self.val_d[i][-1] == 1])
        n_other = len([1 for i in range(len(self.val_t))
                       if self.val_t[i][-1] == 0])
        self.assertEqual(n_icrs, 920)
        self.assertEqual(n_other, 6794)

        n_icrs = len([1 for i in range(len(self.test_d))
                      if self.test_d[i][-1] == 1])
        n_other = len([1 for i in range(len(self.test_t))
                       if self.test_t[i][-1] == 0])
        self.assertEqual(n_icrs, 871)
        self.assertEqual(n_other, 6850)

    def test_get_item(self):
        idx, context, last_msg, img, label = self.train_d[7027]
        self.assertEqual(idx, 7027)
        self.assertEqual(tuple(context.shape), (768,))
        self.assertEqual(tuple(last_msg.shape), (768,))
        self.assertEqual(tuple(img.shape), (2048,))
        self.assertTrue(isinstance(label, int))
        self.assertTrue(label in (0, 1))
        idx, context, last_msg, img, label = self.train_t[7027]
        self.assertEqual(idx, 7027)
        self.assertEqual(tuple(context.shape), (768,))
        self.assertEqual(tuple(last_msg.shape), (768,))
        self.assertEqual(tuple(img.shape), (2048,))
        self.assertTrue(isinstance(label, int))
        self.assertTrue(label in (0, 1))

        idx, context, last_msg, img, label = self.train_d[9927]
        self.assertEqual(idx, 9927)
        self.assertEqual(tuple(context.shape), (768,))
        self.assertEqual(tuple(last_msg.shape), (768,))
        self.assertEqual(tuple(img.shape), (2048,))
        self.assertTrue(isinstance(label, int))
        self.assertTrue(label in (0, 1))
        idx, context, last_msg, img, label = self.train_t[9927]
        self.assertEqual(idx, 9927)
        self.assertEqual(tuple(context.shape), (768,))
        self.assertEqual(tuple(last_msg.shape), (768,))
        self.assertEqual(tuple(img.shape), (2048,))
        self.assertTrue(isinstance(label, int))
        self.assertTrue(label in (0, 1))

    def test_n_labels(self):
        self.assertEqual(self.train_d.n_labels, 2)
        self.assertEqual(self.val_d.n_labels, 2)
        self.assertEqual(self.test_d.n_labels, 2)
        self.assertEqual(self.train_t.n_labels, 2)
        self.assertEqual(self.val_t.n_labels, 2)
        self.assertEqual(self.test_t.n_labels, 2)

    def test_construct(self):
        # TODO: we are not testing downsample, upsample, only_until_peek,
        # only_with_icrs, and remove_first because they are not used in the
        # paper; future use should implement it
        ...

    def test_get_datapoint(self):
        datapoint = self.train_d._get_datapoint(7975, 0)
        self.assertEqual(datapoint.label, 1)
        datapoint = self.train_d._get_datapoint(7975, 1)
        self.assertEqual(datapoint.label, 0)

        datapoint = self.train_t._get_datapoint(7975, 0)
        self.assertEqual(datapoint.label, 1)
        datapoint = self.train_t._get_datapoint(7975, 1)
        self.assertEqual(datapoint.label, 0)

    def test_get_label(self):
        game = self.train_d.games[7014]
        self.assertEqual(self.train_d.get_label(game, turn=0), 0)
        self.assertEqual(self.train_d.get_label(game, turn=1), 1)
        self.assertEqual(self.train_d.get_label(game, turn=2), 0)
        self.assertEqual(self.train_t.get_label(game, turn=3), 0)
        self.assertEqual(self.train_t.get_label(game, turn=4), 0)
        self.assertEqual(self.train_t.get_label(game, turn=5), 1)
        self.assertEqual(self.train_t.get_label(game, turn=6), 0)

        game = self.val_d.games[7988]
        self.assertEqual(self.val_d.get_label(game, turn=0), 0)
        self.assertEqual(self.val_d.get_label(game, turn=1), 1)
        self.assertEqual(self.val_d.get_label(game, turn=2), 1)
        self.assertEqual(self.val_t.get_label(game, turn=3), 1)
        self.assertEqual(self.val_t.get_label(game, turn=4), 0)
        self.assertEqual(self.val_t.get_label(game, turn=5), 0)
        self.assertEqual(self.val_t.get_label(game, turn=6), 0)


if __name__ == '__main__':
    unittest.main()
