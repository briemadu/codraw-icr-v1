#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/structs/actions.py.
"""

import random
import unittest
from collections import Counter

import json

from icr.structs.actions import ActionSeq, Add, Move, Resize, Flip, Delete
from icr.structs.actions import get_clipart_actions, get_scene_diff_actions
from icr.structs.scenes import Scene
from icr.structs.clipart import ClipArt

CODRAW_PATH = './data/CoDraw-master/dataset/CoDraw_1_0.json'
N_SAMPLE = 15

random.seed(123)


class TestAction(unittest.TestCase):
    """Test the action and subactions classes."""
    def setUp(self):
        self.action1 = Move('dog')
        self.action2 = Move('cat')
        self.action3 = Move('dog')
        self.action4 = Add('ball')
        self.action5 = Add('cat')

    def test_eq(self):
        self.assertTrue(self.action1 == self.action3)
        self.assertFalse(self.action1 == self.action2)
        self.assertFalse(self.action2 == self.action4)
        self.assertFalse(self.action4 == self.action5)


class TestFunctions(unittest.TestCase):
    """Test the two auxiliary functions."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)

    def setUp(self):
        clip1 = 's_4s.png,0,4,0,-10000,-10000,0,0'.split(',')
        self.clip1 = ClipArt(clip1)
        clip2 = 's_4s.png,0,4,0,24,67,0,0'.split(',')
        self.clip2 = ClipArt(clip2)
        clip3 = 's_4s.png,0,4,0,24,67,1,0'.split(',')
        self.clip3 = ClipArt(clip3)
        clip4 = 's_4s.png,0,4,0,24,67,1,1'.split(',')
        self.clip4 = ClipArt(clip4)
        clip5 = 's_4s.png,0,4,0,27,67,0,0'.split(',')
        self.clip5 = ClipArt(clip5)
        clip6 = 's_4s.png,0,4,0,24,61,0,0'.split(',')
        self.clip6 = ClipArt(clip6)
        clip7 = 's_4s.png,0,4,0,24,61,0,1'.split(',')
        self.clip7 = ClipArt(clip7)

    def test_get_clipart_actions(self):
        # object does not exist
        actions = get_clipart_actions(self.clip1, self.clip1)
        self.assertEqual(actions, [])
        # object was added
        actions = get_clipart_actions(self.clip1, self.clip2)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Add)
        # object was added (and also flipped/resised, but should not matter)
        actions = get_clipart_actions(self.clip1, self.clip4)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Add)
        # object was deleted
        actions = get_clipart_actions(self.clip4, self.clip1)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Delete)
        # object was only moved (only x and only y)
        actions = get_clipart_actions(self.clip2, self.clip5)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Move)
        actions = get_clipart_actions(self.clip2, self.clip6)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Move)
        # object was only flipped
        actions = get_clipart_actions(self.clip3, self.clip4)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Flip)
        # object was only resized
        actions = get_clipart_actions(self.clip3, self.clip2)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], Resize)
        # object was moved and flipped
        actions = get_clipart_actions(self.clip7, self.clip2)
        self.assertEqual(len(actions), 2)
        self.assertTrue(any(isinstance(x, Flip) for x in actions))
        self.assertTrue(any(isinstance(x, Move) for x in actions))
        # object was flipped and resized
        actions = get_clipart_actions(self.clip4, self.clip2)
        self.assertEqual(len(actions), 2)
        self.assertTrue(any(isinstance(x, Flip) for x in actions))
        self.assertTrue(any(isinstance(x, Resize) for x in actions))
        # object was moved and resized
        actions = get_clipart_actions(self.clip7, self.clip4)
        self.assertEqual(len(actions), 2)
        self.assertTrue(any(isinstance(x, Move) for x in actions))
        self.assertTrue(any(isinstance(x, Resize) for x in actions))
        # object was moved, flipped and resized
        actions = get_clipart_actions(self.clip6, self.clip4)
        self.assertEqual(len(actions), 3)
        self.assertTrue(any(isinstance(x, Move) for x in actions))
        self.assertTrue(any(isinstance(x, Resize) for x in actions))
        self.assertTrue(any(isinstance(x, Flip) for x in actions))

    def test_get_scene_diff_actions(self):
        dialogue = self.codraw['data']['train_07017']['dialog']
        before = Scene('')
        after = Scene(dialogue[0]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 7)
        self.assertTrue(all(isinstance(x, Add) for x in actions))
        before = Scene(dialogue[0]['abs_d'])
        after = Scene(dialogue[1]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 3)
        self.assertTrue(all(isinstance(x, Move) for x in actions))
        before = Scene(dialogue[1]['abs_d'])
        after = Scene(dialogue[2]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 7)
        self.assertEqual(sum(isinstance(x, Move) for x in actions), 6)
        self.assertEqual(sum(isinstance(x, Flip) for x in actions), 1)

        dialogue = self.codraw['data']['train_00377']['dialog']
        before = Scene('')
        after = Scene(dialogue[0]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 0)
        before = Scene(dialogue[0]['abs_d'])
        after = Scene(dialogue[1]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 1)
        self.assertTrue(all(isinstance(x, Add) for x in actions))
        before = Scene(dialogue[1]['abs_d'])
        after = Scene(dialogue[2]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 3)
        self.assertTrue(all(isinstance(x, Add) for x in actions))

        dialogue = self.codraw['data']['train_05664']['dialog']
        before = Scene('')
        after = Scene(dialogue[0]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(isinstance(x, Add) for x in actions))
        before = Scene(dialogue[0]['abs_d'])
        after = Scene(dialogue[1]['abs_d'])
        actions = get_scene_diff_actions(before, after)
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(isinstance(x, Delete) for x in actions))


def create_actionseq_types(codraw, name):
    """Create the two action sequence variations for testing.
        - build from the full game dictionary
        - build incrementally turn by turn
    """
    seq = codraw['data'][name]['dialog']
    # test dialogue built at once with peek
    seq_built = ActionSeq(sequence=seq, include_peek=True)
    # test dialogue built at once without peek
    seq_built_np = ActionSeq(sequence=seq, include_peek=False)

    peek_turn = None
    # test dialogue built incrementaly with peek
    seq_inc = ActionSeq()
    previous = Scene('')
    for t, turn in enumerate(seq):
        current = Scene(turn['abs_d'])
        peeked = 'peeked' in turn
        seq_inc.add_turn(t, previous, current, peeked)
        previous = current
        if 'peeked' in turn:
            peek_turn = t
    # test dialogue built incrementaly without pee
    seq_inc_np = ActionSeq()
    previous = Scene('')
    for t, turn in enumerate(seq):
        if 'peeked' in turn:
            break
        current = Scene(turn['abs_d'])
        peeked = True if 'peeked' in turn else False
        seq_inc_np.add_turn(t, previous, current, peeked)
        previous = current

    return seq_built, seq_built_np, seq_inc, seq_inc_np, peek_turn


class TestActionSeq(unittest.TestCase):
    """Test tsequence of actions class."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)

    def setUp(self):
        # create a sample of game sequences to be tested
        fixed = ['train_00377', 'train_00551', 'train_00791', 'train_07017']
        idxs = random.sample(self.codraw['data'].keys(), N_SAMPLE) + fixed
        self.sample = {idx: create_actionseq_types(self.codraw, idx)
                       for idx in idxs}

    def test_build(self):
        # not necessary because following tests check the built structure
        pass

    def test_add_turn(self):
        # not necessary because following tests check the built structure
        seq = ActionSeq()
        previous = Scene('')
        turn = self.codraw['data']['train_06232']['dialog'][0]
        current = Scene(turn['abs_d'])
        seq.add_turn(0, previous, current, False)
        with self.assertRaises(AssertionError):
            seq.add_turn(2, previous, current, False)
        seq.add_turn(1, previous, current, True)
        self.assertEqual(seq.peek_turn, 1)
        self.assertEqual(len(seq.seq), 2)

    def test_peek_turn(self):
        for game_seqs in self.sample.values():
            built, built_np, inc, inc_np, peek_turn = game_seqs
            self.assertEqual(built.peek_turn, peek_turn)
            self.assertEqual(built_np.peek_turn, None)
            self.assertEqual(inc.peek_turn, peek_turn)
            self.assertEqual(inc_np.peek_turn, None)
            # test the setter
            built_np.peek_turn = 4
            self.assertEqual(built_np.peek_turn, 4)
            with self.assertRaises(ValueError):
                inc_np.peek_turn = 120

    def test_n_turns(self):
        for name, game_seq in self.sample.items():
            built, built_np, inc, inc_np, peek_turn = game_seq
            orig_dialogue = self.codraw['data'][name]['dialog']
            n = len(orig_dialogue)
            until_peek = n if not peek_turn else peek_turn
            self.assertEqual(built.n_turns, n)
            self.assertEqual(built_np.n_turns, until_peek)
            self.assertEqual(inc.n_turns, n)
            self.assertEqual(inc_np.n_turns, until_peek)

    def test_n_actions_per_turn(self):
        built, built_np, inc, inc_np, _ = self.sample['train_00377']
        turns = [0, 1, 3, 5, 2, 8, 5, 0, 14]
        turns_before_peek = [0, 1, 3, 5, 2, 8, 5, 0]
        self.assertEqual(built.n_actions_per_turn(), turns)
        self.assertEqual(built_np.n_actions_per_turn(), turns_before_peek)
        self.assertEqual(inc.n_actions_per_turn(), turns)
        self.assertEqual(inc_np.n_actions_per_turn(), turns_before_peek)

        built, built_np, inc, inc_np, _ = self.sample['train_07017']
        turns = [7, 3, 7, 1, 2, 1, 0, 0, 0]
        turns_before_peek = [7, 3, 7, 1]
        self.assertEqual(built.n_actions_per_turn(), turns)
        self.assertEqual(built_np.n_actions_per_turn(), turns_before_peek)
        self.assertEqual(inc.n_actions_per_turn(), turns)
        self.assertEqual(inc_np.n_actions_per_turn(), turns_before_peek)

    def test_n_actions(self):
        built, built_np, inc, inc_np, _ = self.sample['train_00377']
        self.assertEqual(built.n_actions, 38)
        self.assertEqual(built_np.n_actions, 24)
        self.assertEqual(inc.n_actions, 38)
        self.assertEqual(inc_np.n_actions, 24)

        built, built_np, inc, inc_np, _ = self.sample['train_00551']
        self.assertEqual(built.n_actions, 8)
        self.assertEqual(built_np.n_actions, 8)
        self.assertEqual(inc.n_actions, 8)
        self.assertEqual(inc_np.n_actions, 8)

        built, built_np, inc, inc_np, _ = self.sample['train_00791']
        self.assertEqual(built.n_actions, 7)
        self.assertEqual(built_np.n_actions, 7)
        self.assertEqual(inc.n_actions, 7)
        self.assertEqual(inc_np.n_actions, 7)

        built, built_np, inc, inc_np, _ = self.sample['train_07017']
        self.assertEqual(built.n_actions, 21)
        self.assertEqual(built_np.n_actions, 18)
        self.assertEqual(inc.n_actions, 21)
        self.assertEqual(inc_np.n_actions, 18)

    def test_n_actions_before_peek(self):
        built, built_np, inc, inc_np, _ = self.sample['train_00377']
        self.assertEqual(built.n_actions_before_peek, 24)
        self.assertEqual(built_np.n_actions_before_peek, 24)
        self.assertEqual(inc.n_actions_before_peek, 24)
        self.assertEqual(inc_np.n_actions_before_peek, 24)

        built, built_np, inc, inc_np, _ = self.sample['train_00551']
        self.assertEqual(built.n_actions_before_peek, 8)
        self.assertEqual(built_np.n_actions_before_peek, 8)
        self.assertEqual(inc.n_actions_before_peek, 8)
        self.assertEqual(inc_np.n_actions_before_peek, 8)

        built, built_np, inc, inc_np, _ = self.sample['train_00791']
        self.assertEqual(built.n_actions_before_peek, 7)
        self.assertEqual(built_np.n_actions_before_peek, 7)
        self.assertEqual(inc.n_actions_before_peek, 7)
        self.assertEqual(inc_np.n_actions_before_peek, 7)

        built, built_np, inc, inc_np, _ = self.sample['train_07017']
        self.assertEqual(built.n_actions_before_peek, 18)
        self.assertEqual(built_np.n_actions_before_peek, 18)
        self.assertEqual(inc.n_actions_before_peek, 18)
        self.assertEqual(inc_np.n_actions_before_peek, 18)

    def test_n_actions_after_peek(self):
        built, built_np, inc, inc_np, _ = self.sample['train_00377']
        self.assertEqual(built.n_actions_after_peek, 14)
        self.assertEqual(built_np.n_actions_after_peek, None)
        self.assertEqual(inc.n_actions_after_peek, 14)
        self.assertEqual(inc_np.n_actions_after_peek, None)

        built, built_np, inc, inc_np, _ = self.sample['train_00551']
        self.assertEqual(built.n_actions_after_peek, None)
        self.assertEqual(built_np.n_actions_after_peek, None)
        self.assertEqual(inc.n_actions_after_peek, None)
        self.assertEqual(inc_np.n_actions_after_peek, None)

        built, built_np, inc, inc_np, _ = self.sample['train_00791']
        self.assertEqual(built.n_actions_after_peek, 0)
        self.assertEqual(built_np.n_actions_after_peek, None)
        self.assertEqual(inc.n_actions_after_peek, 0)
        self.assertEqual(inc_np.n_actions_after_peek, None)

        built, built_np, inc, inc_np, _ = self.sample['train_07017']
        self.assertEqual(built.n_actions_after_peek, 3)
        self.assertEqual(built_np.n_actions_after_peek, None)
        self.assertEqual(inc.n_actions_after_peek, 3)
        self.assertEqual(inc_np.n_actions_after_peek, None)

    def test_count_actions(self):

        built, built_np, inc, inc_np, _ = self.sample['train_07017']
        all_acts = Counter({'added': 7, 'moved': 13, 'flipped': 1, 'none': 3})
        before = Counter({'added': 7, 'moved': 10, 'flipped': 1})
        after = Counter({'moved': 3, 'none': 3})
        empty = Counter()

        self.assertEqual(built.count_actions(), all_acts)
        self.assertEqual(built_np.count_actions(), before)
        self.assertEqual(inc.count_actions(), all_acts)
        self.assertEqual(inc_np.count_actions(), before)

        self.assertEqual(built.count_actions(only_before_peek=True), before)
        self.assertEqual(built_np.count_actions(only_before_peek=True), before)
        self.assertEqual(inc.count_actions(only_before_peek=True), before)
        self.assertEqual(inc_np.count_actions(only_before_peek=True), before)

        self.assertEqual(built.count_actions(only_after_peek=True), after)
        self.assertEqual(built_np.count_actions(only_after_peek=True), empty)
        self.assertEqual(inc.count_actions(only_after_peek=True), after)
        self.assertEqual(inc_np.count_actions(only_after_peek=True), empty)

        built, built_np, inc, inc_np, _ = self.sample['train_00551']
        all_acts = Counter({'added': 8, 'none': 5})
        before = Counter({'added': 8, 'none': 5})
        after = Counter()

        self.assertEqual(built.count_actions(), all_acts)
        self.assertEqual(built_np.count_actions(), before)
        self.assertEqual(inc.count_actions(), all_acts)
        self.assertEqual(inc_np.count_actions(), before)

        self.assertEqual(built.count_actions(only_before_peek=True), before)
        self.assertEqual(built_np.count_actions(only_before_peek=True), before)
        self.assertEqual(inc.count_actions(only_before_peek=True), before)
        self.assertEqual(inc_np.count_actions(only_before_peek=True), before)

        self.assertEqual(built.count_actions(only_after_peek=True), after)
        self.assertEqual(built_np.count_actions(only_after_peek=True), empty)
        self.assertEqual(inc.count_actions(only_after_peek=True), after)
        self.assertEqual(inc_np.count_actions(only_after_peek=True), empty)

        built, built_np, inc, inc_np, _ = self.sample['train_00791']
        all_acts = Counter({'added': 6, 'moved': 1, 'none': 3})
        before = Counter({'added': 6, 'moved': 1, 'none': 2})
        after = Counter({'none': 1})

        self.assertEqual(built.count_actions(), all_acts)
        self.assertEqual(built_np.count_actions(), before)
        self.assertEqual(inc.count_actions(), all_acts)
        self.assertEqual(inc_np.count_actions(), before)

        self.assertEqual(built.count_actions(only_before_peek=True), before)
        self.assertEqual(built_np.count_actions(only_before_peek=True), before)
        self.assertEqual(inc.count_actions(only_before_peek=True), before)
        self.assertEqual(inc_np.count_actions(only_before_peek=True), before)

        self.assertEqual(built.count_actions(only_after_peek=True), after)
        self.assertEqual(built_np.count_actions(only_after_peek=True), empty)
        self.assertEqual(inc.count_actions(only_after_peek=True), after)
        self.assertEqual(inc_np.count_actions(only_after_peek=True), empty)

        built, built_np, inc, inc_np, _ = self.sample['train_00377']
        all_acts = Counter({'added': 9, 'moved': 21, 'resized': 8, 'none': 2})
        before = Counter({'added': 9, 'moved': 14, 'resized': 1, 'none': 2})
        after = Counter({'moved': 7, 'resized': 7})

        self.assertEqual(built.count_actions(), all_acts)
        self.assertEqual(built_np.count_actions(), before)
        self.assertEqual(inc.count_actions(), all_acts)
        self.assertEqual(inc_np.count_actions(), before)

        self.assertEqual(built.count_actions(only_before_peek=True), before)
        self.assertEqual(built_np.count_actions(only_before_peek=True), before)
        self.assertEqual(inc.count_actions(only_before_peek=True), before)
        self.assertEqual(inc_np.count_actions(only_before_peek=True), before)

        self.assertEqual(built.count_actions(only_after_peek=True), after)
        self.assertEqual(built_np.count_actions(only_after_peek=True), empty)
        self.assertEqual(inc.count_actions(only_after_peek=True), after)
        self.assertEqual(inc_np.count_actions(only_after_peek=True), empty)

    def test_get_edits_track(self):
        for game_seq in self.sample.values():
            built, built_np, inc, inc_np, _ = game_seq
            edit_seq = built.get_edits_track()
            n_edits = sum([len(seq.edits) for seq in edit_seq.values()])
            self.assertEqual(n_edits + len(edit_seq), built.n_actions)

            edit_seq = built_np.get_edits_track()
            n_edits = sum([len(seq.edits) for seq in edit_seq.values()])
            self.assertEqual(n_edits + len(edit_seq), built_np.n_actions)

            edit_seq = inc.get_edits_track()
            n_edits = sum([len(seq.edits) for seq in edit_seq.values()])
            self.assertEqual(n_edits + len(edit_seq), inc.n_actions)

            edit_seq = inc_np.get_edits_track()
            n_edits = sum([len(seq.edits) for seq in edit_seq.values()])
            self.assertEqual(n_edits + len(edit_seq), inc_np.n_actions)

    def test_get_distance_to_final(self):
        built, built_np, inc, inc_np, _ = self.sample['train_00377']

        dists = {'girl sad arms_right': 7, 'table': 6,
                 'boy scared arms_right': 6, 'fire': 6, 'grill': 5, 'pizza': 5,
                 'hamburger': 5, 'tree': 4, 'bear': 3}
        self.assertEqual(built.get_distance_to_final(), dists)
        self.assertEqual(inc.get_distance_to_final(), dists)

        dists = {'girl sad arms_right': 4, 'table': 4,
                 'boy scared arms_right': 4, 'fire': 3, 'grill': 2, 'pizza': 3,
                 'hamburger': 3, 'tree': 0, 'bear': 1}
        self.assertEqual(built_np.get_distance_to_final(), dists)
        self.assertEqual(inc_np.get_distance_to_final(), dists)


if __name__ == '__main__':
    unittest.main()
