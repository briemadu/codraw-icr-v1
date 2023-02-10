#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to represent a clipart in a CoDraw scene (from Abstract Scenes).
"""

from typing import Any, List

from icr.structs.dataconf import file2obj, clipart_types, NON_EXISTING


class ClipArt:
    """
    Represent a ClipArt with position (x, y), flip, size and whether it
    exists in a scene.
    """
    def __init__(self, attributes: List[str]):
        """Instantiate a clipart object from its string representation.

        Args:
            attributes (list): a list containing the 8 attributes of a clipart
                as strings in this order:
                  - png name: the <name>.png from AbstractScenes cliparts
                  - idx in the scene: integer for CoDraw order
                  - clipart obj idx: clipart id
                  - clipart type idx: clipart group id
                  - x position: x coordinate of clipart in scene
                  - y position: y coordinate of clipart in scene
                  - z value: size id of clipart in scene
                  - flip value: direction id of clipart in scene
        """
        png, local_idx, cobj, ctype, x, y, z, flip = attributes
        self.png = png
        self.local_idx = int(local_idx)
        self.clipart_obj = int(cobj)
        self.clipart_type = int(ctype)
        self.x = float(x)
        self.y = float(y)
        self.z = int(z)
        self.flip = int(flip)
        self._run_checks()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ClipArt):
            return False
        is_same = (self.png == other.png
                   and self.local_idx == other.local_idx
                   and self.clipart_obj == other.clipart_obj
                   and self.clipart_type == other.clipart_type
                   and self.x == other.x
                   and self.y == other.y
                   and self.z == other.z
                   and self.flip == other.flip)
        return is_same

    def _run_checks(self) -> None:
        """Run sanity checks upon creation of the object."""
        c_obj = int(self.png.split('_')[1].strip('.png')[:-1])
        c_type = self.png.split('_')[0]
        err = 'Clipart attributes do not match the AbstractScenes convention.'
        assert c_obj == self.clipart_obj, err
        assert self.z in (0, 1, 2), err
        assert self.flip in (0, 1), err
        assert NON_EXISTING <= self.x, err
        assert NON_EXISTING <= self.y, err
        assert clipart_types[c_type] == self.clipart_type, err

    @property
    def exists(self) -> bool:
        """Return True if the clipart is shown in the scene."""
        # according to
        # https://github.com/facebookresearch/CoDraw/blob/b209770a327f48fdd768724bbcf2783897b0c7fb/js/Abs_util.js#L80
        # if either one or the other is -10000, the clipart is not included
        if self.x == NON_EXISTING or self.y == NON_EXISTING:
            return False
        return True

    @property
    def name(self) -> str:
        """Get the manually created clipart name."""
        return file2obj[self.png]
