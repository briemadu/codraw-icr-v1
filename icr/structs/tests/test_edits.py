#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/structs/edits.py.
"""

import unittest

from icr.structs.edits import EditTrack


class TestEditTrack(unittest.TestCase):
    """Test the edit tracker class."""
    def setUp(self):
        self.edit1 = EditTrack('dog', 1)
        self.edit1.add_edit(2, 'flip')
        self.edit1.add_edit(4, 'move')
        self.edit2 = EditTrack('dog', 0)
        self.edit2.add_edit(5, 'resize')
        self.edit2.add_edit(6, 'delete')
        self.edit2.add_edit(9, 'add')
        self.edit3 = EditTrack('dog', 10)

    def test_edits(self):
        self.assertEqual(self.edit1.edits, ['flip', 'move'])
        self.assertEqual(self.edit2.edits, ['resize', 'delete', 'add'])
        self.assertEqual(self.edit3.edits, [])

    def test_edit_turns(self):
        self.assertEqual(self.edit1.edit_turns, [2, 4])
        self.assertEqual(self.edit2.edit_turns, [5, 6, 9])
        self.assertEqual(self.edit3.edit_turns, [])

    def test_add_edit(self):
        self.assertEqual(self.edit3.edits, [])
        self.edit3.add_edit(11, 'move')
        self.assertEqual(self.edit3.edits, ['move'])
        self.assertEqual(self.edit3.edit_turns, [11])
        self.edit3.add_edit(15, 'move')
        self.assertEqual(self.edit3.edits, ['move', 'move'])
        self.assertEqual(self.edit3.edit_turns, [11, 15])
        with self.assertRaises(AssertionError):
            self.edit3.add_edit(2, 'flip')

    def test_distance(self):
        self.assertEqual(self.edit1.distance, 3)
        self.assertEqual(self.edit2.distance, 9)
        self.assertEqual(self.edit3.distance, 0)

    def test_edited(self):
        self.assertTrue(self.edit1.edited)
        self.assertTrue(self.edit2.edited)
        self.assertFalse(self.edit3.edited)


if __name__ == '__main__':
    unittest.main()
