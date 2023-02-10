#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to represent a sequence of edits for one clipart in CoDraw.
"""

from typing import List


class EditTrack():
    """Represent the track of edits for one clipart."""
    def __init__(self, clipart: str, added_at: int):
        self.clipart = clipart
        self.added_at = added_at
        self._edits: List[str] = []
        self._edit_turns: List[int] = []

    @property
    def edits(self) -> List[str]:
        """The sequence of edit types."""
        return self._edits

    @property
    def edit_turns(self) -> List[int]:
        """The sequence of turns with edits."""
        return self._edit_turns

    def add_edit(self, turn: int, edit_type: str) -> None:
        """Append a new edit."""
        assert turn > self.added_at, "Clipart cannot be edited before add."
        self._edit_turns.append(turn)
        self._edits.append(edit_type)

    @property
    def distance(self) -> int:
        """Return the distance from addition to the final state."""
        if not self.edits:
            return 0
        return self.edit_turns[-1] - self.added_at

    @property
    def edited(self) -> bool:
        """Whether there has been at least one edit."""
        return bool(self.edits)

    def print_track(self) -> None:
        """Print the sequence of actions upon this clipart."""
        print(f'added at turn {self.added_at}')
        for t, act in zip(self.edit_turns, self.edits):
            print(f'{act} at turn {t}')
