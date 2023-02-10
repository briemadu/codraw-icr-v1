#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/structs/scenes.py.
"""

import random
import unittest

import json

from icr.structs.scenes import Scene, SceneSeq

CODRAW_PATH = './data/CoDraw-master/dataset/CoDraw_1_0.json'
N_SAMPLE = 15

random.seed(123)


def create_sceneseq_types(codraw, name):
    """Create the two scene sequence variations for testing.
        - build from the full game dictionary
        - build incrementally turn by turn
    """
    seq = codraw['data'][name]['dialog']
    img_id = codraw['data'][name]['image_id']
    orig_scene_str = codraw['data'][name]['abs_t']
    # test dialogue built at once with peek
    seq_built = SceneSeq(img_id, orig_scene_str, sequence=seq)
    # test dialogue built incrementaly without peek
    seq_inc = SceneSeq(img_id, orig_scene_str)
    for i, turn in enumerate(seq):
        seq_inc.add_scene(i, turn)
    return seq_built, seq_inc


class TestScene(unittest.TestCase):
    """Test the scene class."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)

    def setUp(self):
        scene1 = self.codraw['data']['train_06232']['dialog'][6]['abs_d']
        self.scene1 = Scene(scene1)
        scene2 = self.codraw['data']['train_06242']['dialog'][1]['abs_d']
        self.scene2 = Scene(scene2)
        scene3 = self.codraw['data']['train_06292']['abs_t']
        self.scene3 = Scene(scene3)
        scene4 = self.codraw['data']['train_06242']['dialog'][0]['abs_d']
        self.scene4 = Scene(scene4)
        scene5 = self.codraw['data']['train_06701']['dialog'][0]['abs_d']
        self.scene5 = Scene(scene5)

    def test_get_cliparts(self):
        # indirectly tested by the count methods
        pass

    def test_n_present_cliparts(self):
        self.assertEqual(self.scene1.n_present_cliparts, 7)
        self.assertEqual(self.scene2.n_present_cliparts, 1)
        self.assertEqual(self.scene3.n_present_cliparts, 7)
        self.assertEqual(self.scene4.n_present_cliparts, 0)
        self.assertEqual(self.scene5.n_present_cliparts, 0)

    def test_n_total_cliparts(self):
        self.assertEqual(self.scene1.n_total_cliparts, 28)
        self.assertEqual(self.scene2.n_total_cliparts, 28)
        self.assertEqual(self.scene3.n_total_cliparts, 7)
        self.assertEqual(self.scene4.n_total_cliparts, 0)
        self.assertEqual(self.scene5.n_total_cliparts, 28)

    def test_is_empty(self):
        self.assertFalse(self.scene1.is_empty)
        self.assertFalse(self.scene2.is_empty)
        self.assertFalse(self.scene3.is_empty)
        self.assertTrue(self.scene4.is_empty)
        self.assertTrue(self.scene5.is_empty)


class TestSceneSeq(unittest.TestCase):
    """Test the scene sequence class."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)

    def setUp(self):
        # create a sample of dialogues to be tested
        fixed = ['train_00377',  # icrs only before peek
                 'train_06232',  # icrs only before peek
                 'train_09841',  # icrs before and after peek
                 'train_00551',  # no peek, no icrs
                 'train_00791',  # no real turns after peek
                 'train_07017',  # icrs also after peek
                 'train_00776',  # icrs only after peek
                 ]
        idxs = random.sample(self.codraw['data'].keys(), N_SAMPLE) + fixed
        self.sample = {idx: create_sceneseq_types(self.codraw, idx)
                       for idx in idxs}

    def test_build(self):
        for name, (built, inc) in self.sample.items():
            game = self.codraw['data'][name]
            self.assertEqual(built.orig.str, game['abs_t'].strip(','))
            self.assertEqual(inc.orig.str, game['abs_t'].strip(','))
            turns = game['dialog']
            self.assertEqual(len(built.seq), len(turns))
            self.assertEqual(len(inc.seq), len(turns))
            self.assertEqual(built.seq[0].str, turns[0]['abs_d'].strip(','))
            self.assertEqual(built.seq[-1].str, turns[-1]['abs_d'].strip(','))
            self.assertEqual(inc.seq[0].str, turns[0]['abs_d'].strip(','))
            self.assertEqual(inc.seq[-1].str, turns[-1]['abs_d'].strip(','))

    def test_add_scene(self):
        game = self.codraw['data']['train_06232']
        turn = game['dialog'][0]
        empty_seq = SceneSeq('123', game['abs_t'])
        empty_seq.add_scene(0, turn)
        with self.assertRaises(AssertionError):
            empty_seq.add_scene(3, turn)


if __name__ == '__main__':
    unittest.main()
