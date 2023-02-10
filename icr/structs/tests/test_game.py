#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/structs/game.py.
"""

import unittest

import json
import numpy as np

from icr.structs.game import Game

CODRAW_PATH = './data/CoDraw-master/dataset/CoDraw_1_0.json'


class TestGame(unittest.TestCase):
    """Test the game class."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)

    def setUp(self):
        name = 'train_00377'
        data = self.codraw['data'][name]
        self.game1 = Game(name, data, set())

    def test_build(self):
        pass

    def test_fill_none_scores(self):
        self.game1.scores = [1, None, 5, 6, 8, None, 30]
        filled = [1, 1, 5, 6, 8, 8, 30]
        self.assertEqual(self.game1._fill_none_scores(), filled)
        self.game1.scores = [None, None, None, 6, 8, None]
        filled = [0, 0, 0, 6, 8, 8]
        self.assertEqual(self.game1._fill_none_scores(), filled)

    def test_score_diffs(self):
        self.game1.scores = [1, 1, 5, 6, 8, 10, 30]
        diffs = np.array([1, 0, 4, 1, 2, 2, 20])
        np.testing.assert_array_equal(self.game1.score_diffs, diffs)
        self.game1.scores = [1.5, 1.7, 1.7, 1.7, 2.7, 3, 3.9]
        diffs = np.array([1.5, 0.2, 0.0, 0.0, 1.0, 0.3, 0.9])
        np.testing.assert_allclose(self.game1.score_diffs, diffs, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
