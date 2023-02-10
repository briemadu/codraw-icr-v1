#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/structs/clipart.py.
"""

import unittest

from icr.structs.clipart import ClipArt


class TestClipArt(unittest.TestCase):
    """Test the clipart class."""
    def setUp(self):
        # clipart strings from CoDraw train_00377
        # non-existing clipart
        clip1 = 's_4s.png,0,4,0,-10000,-10000,0,0'.split(',')
        # modify the position to capture strange cases where the JSON file
        # has cliparts with just one position being out of scene
        clip2 = 't_7s.png,25,7,7,-10000,54,0,0'.split(',')
        clip3 = 't_7s.png,25,7,7,54,-10000,0,0'.split(',')
        # existing cliparts
        clip4 = 'p_3s.png,0,3,1,63,349,0,0'.split(',')
        clip5 = 'e_1s.png,6,1,6,323,286,0,1'.split(',')
        clip6 = 't_14s.png,8,14,7,435,262,2,0'.split(',')

        self.clip1 = ClipArt(clip1)
        self.clip2 = ClipArt(clip2)
        self.clip3 = ClipArt(clip3)
        self.clip4 = ClipArt(clip4)
        self.clip5 = ClipArt(clip5)
        self.clip6 = ClipArt(clip6)

    def test_exists(self):
        self.assertFalse(self.clip1.exists)
        self.assertFalse(self.clip2.exists)
        self.assertFalse(self.clip3.exists)
        self.assertTrue(self.clip4.exists)
        self.assertTrue(self.clip5.exists)
        self.assertTrue(self.clip6.exists)

    def test_eq(self):
        self.assertTrue(self.clip1 == self.clip1)
        self.assertTrue(self.clip6 == self.clip6)
        self.assertFalse(self.clip1 == self.clip2)
        self.assertFalse(self.clip2 == self.clip3)
        self.assertFalse(self.clip3 == self.clip4)


if __name__ == '__main__':
    unittest.main()
