#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes to represent a scene and a scene sequence in CoDraw.
"""

from typing import List, Tuple

from icr.structs.clipart import ClipArt
from icr.structs.dataconf import N_OBJ_ATTRIBUTES as N
from icr.structs.dataconf import NON_EXISTING


class Scene:
    """Represent an AbstractScene."""
    def __init__(self, scene: str):
        # remove the unnecessary final comma
        self.str = scene.strip(',')

    def get_cliparts(self) -> List[ClipArt]:
        """Return a list of cliparts (ClipArt objects) in the scene."""
        if self.str == '':
            return []
        total, *objs = self.str.split(',')
        assert len(objs) % N == 0
        obj_list = [objs[i: i+N] for i in range(0, len(objs), N)]
        cliparts = [ClipArt(attributes) for attributes in obj_list]
        assert int(total) == len(cliparts)
        return cliparts

    def list_cliparts(self) -> None:
        """Prints the state of objects in the gallery."""
        for obj in self.get_cliparts():
            print(f'{obj.name}:\t x: {obj.x}, y: {obj.y}')
            print(f'\t size: {obj.z}, flip: {obj.flip}')

    def list_existing_cliparts(self) -> None:
        """Prints the state of objects in the scene."""
        for obj in self.get_cliparts():
            if obj.exists:
                print(f'{obj.name}:\t x: {obj.x:.2f}, y: {obj.y:.2f}'
                      f'\t size: {obj.z}, flip: {obj.flip}')

    def clipart_positions(self) -> List[Tuple[str, float, float]]:
        """Return a list of (id, x, y) positions for all cliparts."""
        return [(obj.png, obj.x, obj.y) for obj in self.get_cliparts()]

    @property
    def n_present_cliparts(self) -> int:
        """Return the number of objects present in a scene."""
        return len([x for x in self.get_cliparts() if x.exists])

    @property
    def n_total_cliparts(self) -> int:
        """Return the number of cliparts in the scene string."""
        return len(self.get_cliparts())

    @property
    def is_empty(self) -> bool:
        """Check whether the scene has any cliparts."""
        positions = self.clipart_positions()
        for _, x, y in positions:
            # apparently some empty images have a few objects with one
            # coordinate = -10000 and the other valid...
            # we consider cliparts with both coordinates valid, according to
            # https://github.com/facebookresearch/CoDraw/blob/b209770a327f48fdd768724bbcf2783897b0c7fb/js/Abs_util.js#L80
            if int(float(x)) != NON_EXISTING and int(float(y)) != NON_EXISTING:
                return False
        return True


class SceneSeq:
    """Represent a sequence of scenes in a CoDraw game."""
    def __init__(self, img_id: str, orig_scene: str, **kwargs):
        self.img_id = img_id
        self.orig = Scene(orig_scene)
        self.seq: List[Scene] = []
        if 'sequence' in kwargs:
            self.build(kwargs['sequence'])

    def build(self, seq: List[dict]) -> None:
        """Extract and store a full scene sequence at once."""
        for i, turn in enumerate(seq):
            self.add_scene(i, turn)

    def add_scene(self, t: int, turn: dict) -> None:
        """Add one scene (use to build game sequence incrementally)."""
        assert len(self.seq) == t
        self.seq.append(Scene(turn['abs_d']))

    def print_seq(self) -> None:
        """Print the cliparts in each scene in the sequence."""
        for s, scn in enumerate(self.seq):
            print(f'\nTurn {s}\n')
            scn.list_existing_cliparts()
