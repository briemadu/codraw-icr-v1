#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to represent a game in CoDraw.
"""

import sys
sys.path.append('../codrawmodels')
from typing import List, Optional, Set

import numpy as np
import matplotlib.pyplot as plt

from icr.structs.dialogue import Dialogue
from icr.structs.scenes import SceneSeq, Scene
from icr.structs.actions import ActionSeq
from codrawmodels.codraw_data import AbstractScene
from codrawmodels.abs_metric import scene_similarity


class Game:
    """Represent a CoDraw game."""
    def __init__(self, idx: str, game: dict, icrs: Set[str],
                 quick_load: bool = False,
                 include_peek: bool = True):
        self.copy_attributes(idx, quick_load, include_peek)
        self.initialise(game)
        self.build(game, icrs)

    def copy_attributes(self, game_id: str, quick_load: bool,
                        include_peek: bool) -> None:
        """Copy game details."""
        self.quick_load = quick_load
        self.include_peek = include_peek
        self.game_id = game_id
        self.id = int(game_id.split('_')[1])

    def initialise(self, game: dict) -> None:
        """Initialise structures to store the dialogue."""
        self.dialogue = Dialogue()
        self.scenes = SceneSeq(game['image_id'], game['abs_t'])
        self.peek_turn: Optional[int] = None
        self.n_turns = len(game['dialog'])
        if not self.quick_load:
            self.actions = ActionSeq()
            self.scores: List[Optional[float]] = []

    def build(self, game: dict, icrs: Set[str]) -> None:
        """Extract and store dialogue turns and scores."""
        # we could use the build method of each object, but for efficiency
        # we loop just once here as populate them turn by turn incrementally
        turns_data = game['dialog']
        previous = Scene('')
        for t, turn in enumerate(turns_data):
            if 'peeked' in turn and not self.include_peek:
                self.n_turns = t
                break
            if 'peeked' in turn:
                self.peek_turn = t
            # populate dialogue
            self.dialogue.add_turn(t, turn, icrs)
            # populate scene sequence
            self.scenes.add_scene(t, turn)
            if not self.quick_load:
                # populate action sequence
                peek = 'peeked' in turn
                self.actions.add_turn(t, previous, self.scenes.seq[-1], peek)
                previous = self.scenes.seq[-1]
                # populate sequence of scores
                if turn['abs_d']:
                    pred = AbstractScene(turn['abs_d'])
                    target = AbstractScene(self.scenes.orig.str)
                    score = scene_similarity(pred, target)
                    self.scores.append(score)
                else:
                    self.scores.append(None)
        self.check()

    def check(self) -> None:
        """Sanity check that game's structs look fine."""
        assert len(self.dialogue.turns) == self.n_turns
        assert self.dialogue.n_turns == self.n_turns
        assert len(self.scenes.seq) == self.n_turns
        assert self.peek_turn == self.dialogue.peek_turn
        # check that last scene is never an empty string
        assert len(self.scenes.seq[-1].str) > 50
        if self.peek_turn is not None:
            assert len(self.scenes.seq[self.peek_turn].str) > 50
        if not self.quick_load:
            assert self.peek_turn == self.actions.peek_turn
            assert len(self.actions.seq) == self.n_turns
            assert len(self.scores) == self.n_turns
            assert self.scores[-1] is not None
            if self.peek_turn is not None:
                assert self.scores[self.peek_turn - 1] is not None

    def _fill_none_scores(self) -> List[float]:
        """Handle turns in which abs_d is empty."""
        # replacing Nones
        filled_scores: List[float] = []
        for t, score in enumerate(self.scores):
            # if the initial score is None, the scene has not yet been
            # initialized, so we consider it o be 0
            if score is None and t == 0:
                filled_scores.append(0.0)
            elif score is None:
                # we append the a copy of the last score
                # check sanity-check.ipynb for demonstration that this only
                # occurs in the beginning when no action has been done yet
                # so we could as well replace by 0 as above
                filled_scores.append(filled_scores[-1])
            else:
                filled_scores.append(score)
        return filled_scores

    @property
    def score_diffs(self) -> np.array:
        """Return list of score differences between contiguous turns."""
        filled_scores = self._fill_none_scores()
        return np.array(filled_scores) - np.array([0] + filled_scores[:-1])

    def print_steps(self) -> None:
        """Print the game actions"""
        for t in range(self.n_turns):
            is_icr = t in self.dialogue.icr_turns
            peek_str = 'PEEKED!' if self.peek_turn == t else ''
            print(f'\n### {t}, CR={is_icr} {peek_str}')
            for act in self.actions.seq[t]:
                act.print_act()

    def plot_dialogue(self) -> None:
        """Visualisation of a dialogue as a plot."""
        x = [-1] + list(range(self.n_turns))
        # replace None values by 0
        scores = self._fill_none_scores()
        y = [0] + scores
        plt.plot(x, y, linestyle='-')
        plt.xticks(x, labels=['S'] + x[1:])
        plt.yticks(np.arange(0, 6, 0.5))
        plt.ylim((0, 5))

        icr_scores = [self.scores[i] for i in self.dialogue.icr_turns]
        plt.plot(self.dialogue.icr_turns, icr_scores, 'ro', label='iCR turns')

        aux_loop = range(self.n_turns)
        other_turns = [i for i in aux_loop if i not in self.dialogue.icr_turns]
        other_scores = [self.scores[i] for i in other_turns]
        plt.plot(other_turns, other_scores, 'bo', label='')

        if self.dialogue.peek_turn:
            plt.axvline(self.dialogue.peek_turn - 0.5, color='g',
                        linestyle='--', label='peeked')

        plt.xlabel('turn', fontsize=12)
        plt.ylabel('scene similarity score', fontsize=12)
        plt.title(self.game_id)
        plt.legend()
        plt.show()
