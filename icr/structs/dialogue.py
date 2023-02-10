#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to represent a dialogue in CoDraw.
"""

from collections import namedtuple
from typing import List, Optional, Set

Turn = namedtuple("Turn", ["teller", "drawer"])


class Dialogue:
    """Represent a CoDraw dialogue based on its turns."""
    def __init__(self, **kwargs) -> None:
        self._peek_turn: Optional[int] = None
        self.turns: List[Turn] = []
        self.icr_turns: List[int] = []
        if all(key in kwargs for key in ['dialogue', 'icrs', 'include_peek']):
            self.build(kwargs)

    def build(self, kwargs) -> None:
        """Extract and store a full CoDraw dialogue at once."""
        dialogue = kwargs['dialogue']
        icrs = kwargs['icrs']
        self.include_peek = kwargs['include_peek']
        for t, turn in enumerate(dialogue):
            if 'peeked' in turn and not self.include_peek:
                break
            self.add_turn(t, turn, icrs)

    def add_turn(self, t: int, turn: dict, icrs: Set[str]) -> None:
        """Add a new CoDraw turn and checks if it contains an iCR.
        (use to build the dialogue incrementally)
        """
        assert t == self.n_turns
        messages = Turn(teller=turn['msg_t'], drawer=turn['msg_d'])
        self.turns.append(messages)
        if messages.drawer in icrs:
            self.icr_turns.append(t)
        if 'peeked' in turn:
            self.peek_turn = t

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
    def teller_turns(self) -> List[str]:
        """Get list of teller's turns."""
        return [t.teller for t in self.turns]

    @property
    def drawer_turns(self) -> List[str]:
        """Get list of drawer's turns."""
        return [t.drawer for t in self.turns]

    @property
    def n_turns(self) -> int:
        """Get the dialogue length in numbers of turns."""
        return len(self.turns)

    @property
    def n_icrs(self) -> int:
        """Number of turns containing iCRs."""
        return len(self.icr_turns)

    @property
    def icr_turns_before_peek(self) -> List[int]:
        """List of iCR turn ids before the peek action."""
        if self.peek_turn is None:
            return self.icr_turns
        return [t for t in self.icr_turns if t < self.peek_turn]

    @property
    def icr_turns_after_peek(self) -> List[int]:
        """List of iCR turn ids after the peek action."""
        if self.peek_turn is None:
            return []
        return [t for t in self.icr_turns if t >= self.peek_turn]

    @property
    def n_icrs_before_peek(self) -> int:
        """Number of turns containing iCRs before the peek action."""
        return len(self.icr_turns_before_peek)

    @property
    def n_icrs_after_peek(self) -> Optional[int]:
        """Number of turns containing iCRs after the peek action."""
        if self.peek_turn is None:
            return None
        return len(self.icr_turns_after_peek)

    def get_dialogue_string(
            self, sep_t: str = '/T ', sep_d: str = '/D ') -> str:
        """Get full dialogue as a string with defined separators."""
        # TODO: if necessary, we could add the PEEK flag wheren it occurs
        context = " ".join([f'{sep_t}{turn.teller} {sep_d}{turn.drawer} '
                           for turn in self.turns])
        return context

    def print_dialogue(self) -> None:
        """Print all dialogue rounds."""
        print('\n')
        for i in range(self.n_turns):
            if self.peek_turn == i:
                print('\n --- PEEKED --- \n')
            print(f'TELLER: {self.turns[i].teller}')
            print(f'DRAWER: {self.turns[i].drawer}\n')
