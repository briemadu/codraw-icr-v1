#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for icr/structs/dialogue.py.
"""

import random
import unittest

import json
import pandas as pd

from icr.structs.dialogue import Dialogue

CODRAW_PATH = './data/CoDraw-master/dataset/CoDraw_1_0.json'
ANNOTATION_PATH = './data/cr_anno_adjudication/data_adju.tsv'
ICR_COLUMN = 'is CR? v2'
N_SAMPLE = 15

random.seed(123)


def create_dialogue_types(codraw, name, icrs):
    """Create the four dialogue variations for testing.

        - build from the full dialogue dictionary
        - build incrementally turn by turn
        - build from the full dialogue dictionary only up to peek
        - build incrmentally turn by turn only up to peek action
    """
    dialogue = codraw['data'][name]['dialog']
    # test dialogue built at once with peek
    d_built = Dialogue(dialogue=dialogue, icrs=icrs, include_peek=True)
    # test dialogue built at once without peek
    d_built_np = Dialogue(dialogue=dialogue, icrs=icrs, include_peek=False)
    # test dialogue built incrementaly with peek
    d_inc = Dialogue()
    for i, turn in enumerate(dialogue):
        d_inc.add_turn(i, turn, icrs)
    # test dialogue built incrementaly without peek
    d_inc_np = Dialogue()
    for i, turn in enumerate(dialogue):
        if 'peeked' in turn:
            break
        d_inc_np.add_turn(i, turn, icrs)
    peek_turn = [t for t, turn in enumerate(dialogue) if 'peeked' in turn]
    assert len(peek_turn) in (0, 1)
    peek_turn = peek_turn[0] if peek_turn else None

    return d_built, d_built_np, d_inc, d_inc_np, peek_turn


class TestDialogue(unittest.TestCase):
    """Test the dialogue class."""
    @classmethod
    def setUpClass(cls):
        # load CoDraw data
        with open(CODRAW_PATH, 'r') as f:
            cls.codraw = json.load(f)
        # get annotations, here we use SR's labels
        annotated = pd.read_csv(ANNOTATION_PATH, sep='\t')
        pairs = zip(annotated['drawer\'s utterance'], annotated[ICR_COLUMN])
        cls.icrs = {sent for sent, label in pairs if label == 1}

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
        self.sample = {idx: create_dialogue_types(self.codraw, idx, self.icrs)
                       for idx in idxs}

    def test_build(self):
        # not necessary because following tests check the built structure
        pass

    def test_add_turn(self):
        # not necessary because following tests check the built structure
        turn = self.codraw['data']['train_06232']['dialog'][0]
        aux_d = Dialogue()
        aux_d.add_turn(0, turn, self.icrs)
        self.assertEqual(len(aux_d.turns), 1)
        with self.assertRaises(AssertionError):
            aux_d.add_turn(3, turn, self.icrs)
        aux_d.add_turn(1, turn, self.icrs)
        self.assertEqual(len(aux_d.turns), 2)
        aux_turn = {key: value for key, value in turn.items()}
        aux_turn['peeked'] = True
        aux_d.add_turn(2, aux_turn, self.icrs)
        self.assertEqual(len(aux_d.turns), 3)
        self.assertEqual(aux_d.peek_turn, 2)

    def test_peek_turn(self):
        for dialogues in self.sample.values():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            self.assertEqual(d_built.peek_turn, peek_turn)
            self.assertEqual(d_built_np.peek_turn, None)
            self.assertEqual(d_inc.peek_turn, peek_turn)
            self.assertEqual(d_inc_np.peek_turn, None)
            # test the setter
            d_built_np.peek_turn = 2
            self.assertEqual(d_built_np.peek_turn, 2)
            with self.assertRaises(ValueError):
                d_inc_np.peek_turn = 120

    def test_teller_turns(self):
        for name, dialogues in self.sample.items():
            d_built, _, d_inc, _, _ = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            t_msgs = [turn['msg_t'] for turn in orig_dialogue]
            self.assertEqual(d_built.teller_turns, t_msgs)
            self.assertEqual(d_inc.teller_turns, t_msgs)

    def test_drawer_turns(self):
        for name, dialogues in self.sample.items():
            d_built, _, d_inc, _, _ = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            d_msgs = [turn['msg_d'] for turn in orig_dialogue]
            self.assertEqual(d_built.drawer_turns, d_msgs)
            self.assertEqual(d_inc.drawer_turns, d_msgs)

    def test_n_turns(self):
        for name, dialogues in self.sample.items():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            n = len(orig_dialogue)
            until_peek = n if not peek_turn else peek_turn
            self.assertEqual(d_built.n_turns, n)
            self.assertEqual(d_built_np.n_turns, until_peek)
            self.assertEqual(d_inc.n_turns, n)
            self.assertEqual(d_inc_np.n_turns, until_peek)

    def test_n_icrs(self):
        for name, dialogues in self.sample.items():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            icrs_turns = [t for t, turn in enumerate(orig_dialogue)
                          if turn['msg_d'] in self.icrs]
            n = len(icrs_turns)
            until_peek = n
            if peek_turn:
                until_peek = len([t for t in icrs_turns if t < peek_turn])
            self.assertEqual(d_built.n_icrs, n)
            self.assertEqual(d_built_np.n_icrs, until_peek)
            self.assertEqual(d_inc.n_icrs, n)
            self.assertEqual(d_inc_np.n_icrs, until_peek)

    def test_icr_turns_before_peek(self):
        for name, dialogues in self.sample.items():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            n = peek_turn or len(orig_dialogue)
            turns = [t for t in range(n)
                     if orig_dialogue[t]['msg_d'] in self.icrs]
            self.assertEqual(d_built.icr_turns_before_peek, turns)
            self.assertEqual(d_built_np.icr_turns_before_peek, turns)
            self.assertEqual(d_inc.icr_turns_before_peek, turns)
            self.assertEqual(d_inc_np.icr_turns_before_peek, turns)

    def test_icr_turns_after_peek(self):
        for name, dialogues in self.sample.items():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            n1 = peek_turn or len(orig_dialogue)
            n2 = len(orig_dialogue)
            turns = [t for t in range(n1, n2)
                     if orig_dialogue[t]['msg_d'] in self.icrs]
            self.assertEqual(d_built.icr_turns_after_peek, turns)
            self.assertEqual(d_built_np.icr_turns_after_peek, [])
            self.assertEqual(d_inc.icr_turns_after_peek, turns)
            self.assertEqual(d_inc_np.icr_turns_after_peek, [])

    def test_n_icrs_before_peek(self):
        for name, dialogues in self.sample.items():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            n = peek_turn or len(orig_dialogue)
            turns = [t for t in range(n)
                     if orig_dialogue[t]['msg_d'] in self.icrs]
            self.assertEqual(d_built.n_icrs_before_peek, len(turns))
            self.assertEqual(d_built_np.n_icrs_before_peek, len(turns))
            self.assertEqual(d_inc.n_icrs_before_peek, len(turns))
            self.assertEqual(d_inc_np.n_icrs_before_peek, len(turns))

    def test_n_icrs_after_peek(self):
        for name, dialogues in self.sample.items():
            d_built, d_built_np, d_inc, d_inc_np, peek_turn = dialogues
            orig_dialogue = self.codraw['data'][name]['dialog']
            n1 = peek_turn or len(orig_dialogue)
            n2 = len(orig_dialogue)
            turns = [t for t in range(n1, n2)
                     if orig_dialogue[t]['msg_d'] in self.icrs]
            # catch cases where dialogue has no peek turn in the data
            correct_n = len(turns) if peek_turn else None
            self.assertEqual(d_built.n_icrs_after_peek, correct_n)
            self.assertEqual(d_built_np.n_icrs_after_peek, None)
            self.assertEqual(d_inc.n_icrs_after_peek, correct_n)
            self.assertEqual(d_inc_np.n_icrs_after_peek, None)


if __name__ == '__main__':
    unittest.main()
