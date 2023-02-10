#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes to represent an action and an action sequence in CoDraw games.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from icr.structs.clipart import ClipArt
from icr.structs.edits import EditTrack
from icr.structs.scenes import Scene

ADDED: str = 'added'
MOVED: str = 'moved'
FLIPPED: str = 'flipped'
RESIZED: str = 'resized'
DELETED: str = 'deleted'


class Action():
    """Represent one action (add, delete, move, flip or resize)."""
    def __init__(self, clipart: str, action: str):
        self.clipart = clipart
        self.action = action

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.clipart == other.clipart and self.action == other.action

    def print_act(self) -> None:
        """Simple print action +  clipart."""
        print(f'  {self.action} {self.clipart}')


class Add(Action):
    """Represent a clipart that was added."""
    def __init__(self, clipart: str):
        super().__init__(clipart, ADDED)


class Move(Action):
    """Represent a clipart that changed position."""
    def __init__(self, clipart: str):
        super().__init__(clipart, MOVED)


class Resize(Action):
    """Represent a clipart that was resized."""
    def __init__(self, clipart: str):
        super().__init__(clipart, RESIZED)


class Delete(Action):
    """Represent a clipart that was deleted."""
    def __init__(self, clipart: str):
        super().__init__(clipart, DELETED)


class Flip(Action):
    """Represent a clipart that was flipped."""
    def __init__(self, clipart: str):
        super().__init__(clipart, FLIPPED)


def get_clipart_actions(obj_before: ClipArt, obj_after: ClipArt) -> List:
    """Compare state of a clipart with its previous state and return action."""
    assert obj_before.name == obj_after.name
    actions = []
    # CASE A: object does not exist in neither
    if not obj_before.exists and not obj_after.exists:
        return []
    # CASE B: object was deleted
    if obj_before.exists and not obj_after.exists:
        actions.append(Delete(obj_after.name))
    # CASE C: object was added
    #         (!!) we do not care if added objects were flipped/resized
    elif not obj_before.exists and obj_after.exists:
        actions.append(Add(obj_after.name))
    # CASE D: object was edited (moved, flipped and/or resized)
    elif obj_before.exists and obj_after.exists:
        if (obj_after.x != obj_before.x or obj_after.y != obj_before.y):
            actions.append(Move(obj_after.name))
        if obj_after.z != obj_before.z:
            actions.append(Resize(obj_after.name))
        if obj_after.flip != obj_before.flip:
            actions.append(Flip(obj_after.name))
    return actions


def get_scene_diff_actions(scene_before: Scene, scene_after: Scene) -> List:
    """Return a list with the drawer's actions in a dialogue turn."""
    # CASE 1: nothing going on, both scenes are empty
    if scene_before.is_empty and scene_after.is_empty:
        return []

    objs_before = [] if scene_before.is_empty else scene_before.get_cliparts()
    objs_after = [] if scene_after.is_empty else scene_after.get_cliparts()

    # CASE 2: scene gets empty, so all cliparts have been deleted
    if not scene_before.is_empty and scene_after.is_empty:
        actions = [Delete(obj.name) for obj in objs_before if obj.exists]
        # ensure that at least one object was indeed added
        assert len(actions) > 0
        return actions

    # CASE 3: emtpy scene is initialized, thus all objects are newly added
    #        (!!) we do not care if added objects were flipped and resized
    if scene_before.is_empty and not scene_after.is_empty:
        actions = [Add(obj.name) for obj in objs_after if obj.exists]
        # ensure that at least one object was indeed added
        assert len(actions) > 0
        return actions

    # CASE 4: scene is altered
    if not scene_before.is_empty and not scene_after.is_empty:
        actions = []
        # Here we need the assumption that cliparts are unique in a scene!
        # For demonstration, see sanity-checks.ipynb
        dic_before = {obj.name: obj for obj in objs_before}
        dic_after = {obj.name: obj for obj in objs_after}
        # loop over all objects in the current scene and compare their status
        # with the previous scene
        for name, obj_after in dic_after.items():
            obj_before = dic_before[name]
            actions += get_clipart_actions(obj_before, obj_after)
    return actions


class ActionSeq():
    """Represent the sequence of actions in a CoDraw game."""
    def __init__(self, **kwargs) -> None:
        self.seq: List = []
        self._peek_turn: Optional[int] = None
        if 'sequence' in kwargs:
            self.build(kwargs['sequence'], kwargs['include_peek'])

    def build(self, seq: List, include_peek: bool) -> None:
        """Extract and store a full CoDraw dialogue at once."""
        self.include_peek = include_peek
        previous = Scene('')
        for t, turn in enumerate(seq):
            if 'peeked' in turn and not self.include_peek:
                break
            current = Scene(turn['abs_d'])
            peeked = 'peeked' in turn
            self.add_turn(t, previous, current, peeked)
            previous = current

    def add_turn(self, t: int, previous: Scene, current: Scene, peeked: bool):
        """Add a new CoDraw turn and retrieve actions via the diff.
        (use to build the dialogue incrementally)
        """
        assert self.n_turns == t
        if peeked:
            self.peek_turn = t
        turn_actions = get_scene_diff_actions(previous, current)
        self.seq.append(turn_actions)

    @property
    def peek_turn(self) -> Optional[int]:
        """Turn id when peek action occurred."""
        return self._peek_turn

    @peek_turn.setter
    def peek_turn(self, n: int) -> None:
        """Set the turn id when peek action occurred."""
        if n > self.n_turns:
            raise ValueError("Turn does not exist in the dialogue (yet).")
        self._peek_turn = n

    @property
    def n_actions(self) -> int:
        """Total number of actions."""
        return sum([1 for turn in self.seq for act in turn])

    def n_actions_per_turn(self) -> List[int]:
        """Number of actions per game turn."""
        return [len(actions) for actions in self.seq]

    @property
    def n_turns(self) -> int:
        """Number of turns in the game."""
        return len(self.seq)

    @property
    def n_actions_before_peek(self) -> int:
        """Number of actions until the peek."""
        stop = self.peek_turn or self.n_turns
        return sum([1 for turn in self.seq[:stop] for act in turn])

    @property
    def n_actions_after_peek(self) -> Optional[int]:
        """Number of actions after the peek."""
        if not self.peek_turn:
            return None
        return sum([1 for turn in self.seq[self.peek_turn:] for act in turn])

    def count_actions(self, only_before_peek: bool = False,
                      only_after_peek: bool = False) -> Counter:
        """Count each type of action in the game."""
        action_counts: Counter = Counter()
        if not self.peek_turn and only_after_peek:
            return action_counts
        # if no peek action, we loop until the end of the game
        stop = self.peek_turn or self.n_turns
        for t, turn in enumerate(self.seq):
            if t < stop and only_after_peek:
                continue
            if t == stop and only_before_peek:
                break
            if not turn:
                action_counts.update(['none'])
            for action in turn:
                action_counts.update([action.action])
        return action_counts

    def get_edits_track(self) -> Dict[str, EditTrack]:
        """Retrive the track of edits per clipart in the game."""
        # Here we need the assumption that cliparts are unique in a scene!
        # For demonstration, see sanity-checks.ipynb
        edits = {}
        for t, turn in enumerate(self.seq):
            for action in turn:
                clip = action.clipart
                if clip not in edits:
                    assert action.action == ADDED
                    edits[clip] = EditTrack(clip, t)
                else:
                    assert (action.action != ADDED
                            or DELETED in edits[clip].edits)
                    edits[clip].add_edit(t, action.action)
        # make sure that additions + edits match number of actions
        n_edits = sum([len(seq.edits) for seq in edits.values()])
        assert n_edits + len(edits) == self.n_actions
        return edits

    def get_distance_to_final(self) -> Dict[str, int]:
        """Compute how many turns until final positioning of each clipart."""
        seq = self.get_edits_track()
        # a clipart that was added and never moved has distance = 0
        return {clipart: track.distance for clipart, track in seq.items()}

    def print_sequence(self) -> None:
        """Print the sequence of actions."""
        for t, actions in enumerate(self.seq):
            print(f'\n----------  turn {t} ----------')
            if not actions:
                print('  no actions!', '\n')
            else:
                for act in actions:
                    act.print_act()
