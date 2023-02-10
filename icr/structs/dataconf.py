#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoDraw and AbstractScenes data configurations and constants.
We also define here some fixed values in the CoDraw JSON file.
"""

from typing import Dict

# these constants comes from the CoDraw JSON file and the AbstractScene
# documentation about the scene string representation
NON_EXISTING: int = -10000

# each clipart in a CoDraw scene is represented by 8 attributes
# see details in the ClipArt class
N_OBJ_ATTRIBUTES: int = 8

# types from the AbstractScenes documentation
clipart_types: Dict[str, int] = {'s': 0, 'p': 1, 'hb0': 2, 'hb1': 3, 'a': 4,
                                 'c': 5, 'e': 6, 't': 7}

# 56 objects + 2 people in 7 body poses and 5 facial expressions
N_CLIPARTS: int = 126
# maximum number of cliparts in a scene
MAX_CLIPARTS: int = 17
# canvas size
WIDTH: int = 500
HEIGHT: int = 400

# file2obj is a dictionary mapping all AbstractScenes clipart file names to a
# short human-readable description. The names have been created by me just to
# aid inspection and do not necessarily match AbstractScenes or CoDraw's names.
file2obj: Dict[str, str] = {
    # animals
    'a_0s.png': 'bear',
    'a_1s.png': 'cat',
    'a_2s.png': 'dog',
    'a_3s.png': 'duck',
    'a_4s.png': 'owl',
    'a_5s.png': 'snake',
    # background
    'background.png': 'default background with grass and sky',
    # clothing
    'c_0s.png': 'blue hat with star',
    'c_1s.png': 'crown',
    'c_2s.png': 'chef hat',
    'c_3s.png': 'pirate hat',
    'c_4s.png': 'red and gray bonnet',
    'c_5s.png': 'rainbow hat with helix',
    'c_6s.png': 'witch hat',
    'c_7s.png': 'viking helmet',
    'c_8s.png': 'glasses',
    'c_9s.png': 'sunglasses',
    # food
    'e_0s.png': 'pie',
    'e_1s.png': 'pizza',
    'e_2s.png': 'hotdog',
    'e_3s.png': 'catchup',
    'e_4s.png': 'mustard',
    'e_5s.png': 'hamburger',
    'e_6s.png': 'drink',
    # boy (Mike)
    'hb0_0s.png': 'boy angry arms_right',
    'hb0_1s.png': 'boy wide_smile arms_right',
    'hb0_2s.png': 'boy smile arms_right',
    'hb0_3s.png': 'boy sad arms_right',
    'hb0_4s.png': 'boy scared arms_right',
    'hb0_5s.png': 'boy angry arms_up',
    'hb0_6s.png': 'boy wide_smile arms_up',
    'hb0_7s.png': 'boy smile arms_up',
    'hb0_8s.png': 'boy sad arms_up',
    'hb0_9s.png': 'boy scared arms_up',
    'hb0_10s.png': 'boy angry kicking',
    'hb0_11s.png': 'boy wide_smile kicking',
    'hb0_12s.png': 'boy smile kicking',
    'hb0_13s.png': 'boy sad kicking',
    'hb0_14s.png': 'boy scared kicking',
    'hb0_15s.png': 'boy angry running',
    'hb0_16s.png': 'boy wide_smile running',
    'hb0_17s.png': 'boy smile running',
    'hb0_18s.png': 'boy sad running',
    'hb0_19s.png': 'boy scared running',
    'hb0_20s.png': 'boy angry leg_crossed',
    'hb0_21s.png': 'boy wide_smile leg_crossed',
    'hb0_22s.png': 'boy smile leg_crossed',
    'hb0_23s.png': 'boy sad leg_crossed',
    'hb0_24s.png': 'boy scared leg_crossed',
    'hb0_25s.png': 'boy angry sit',
    'hb0_26s.png': 'boy wide_smile sit',
    'hb0_27s.png': 'boy smile sit',
    'hb0_28s.png': 'boy sad sit',
    'hb0_29s.png': 'boy scared sit',
    'hb0_30s.png': 'boy angry wave',
    'hb0_31s.png': 'boy wide_smile wave',
    'hb0_32s.png': 'boy smile wave',
    'hb0_33s.png': 'boy sad wave',
    'hb0_34s.png': 'boy scared wave',
    # girl (Jenny)
    'hb1_0s.png': 'girl angry arms_right',
    'hb1_1s.png': 'girl wide_smile arms_right',
    'hb1_2s.png': 'girl smile arms_right',
    'hb1_3s.png': 'girl sad arms_right',
    'hb1_4s.png': 'girl scared arms_right',
    'hb1_5s.png': 'girl angry arms_up',
    'hb1_6s.png': 'girl wide_smile arms_up',
    'hb1_7s.png': 'girl smile arms_up',
    'hb1_8s.png': 'girl sad arms_up',
    'hb1_9s.png': 'girl scared arms_up',
    'hb1_10s.png': 'girl angry kicking',
    'hb1_11s.png': 'girl wide_smile kicking',
    'hb1_12s.png': 'girl smile kicking',
    'hb1_13s.png': 'girl sad kicking',
    'hb1_14s.png': 'girl scared kicking',
    'hb1_15s.png': 'girl angry running',
    'hb1_16s.png': 'girl wide_smile running',
    'hb1_17s.png': 'girl smile running',
    'hb1_18s.png': 'girl sad running',
    'hb1_19s.png': 'girl scared running',
    'hb1_20s.png': 'girl angry leg_crossed',
    'hb1_21s.png': 'girl wide_smile leg_crossed',
    'hb1_22s.png': 'girl smile leg_crossed',
    'hb1_23s.png': 'girl sad leg_crossed',
    'hb1_24s.png': 'girl scared leg_crossed',
    'hb1_25s.png': 'girl angry sit',
    'hb1_26s.png': 'girl wide_smile sit',
    'hb1_27s.png': 'girl smile sit',
    'hb1_28s.png': 'girl sad sit',
    'hb1_29s.png': 'girl scared sit',
    'hb1_30s.png': 'girl angry wave',
    'hb1_31s.png': 'girl wide_smile wave',
    'hb1_32s.png': 'girl smile wave',
    'hb1_33s.png': 'girl sad wave',
    'hb1_34s.png': 'girl scared wave',
    # large objects
    'p_0s.png': 'bee',
    'p_1s.png': 'slide',
    'p_2s.png': 'sandbox',
    'p_3s.png': 'grill',
    'p_4s.png': 'swing',
    'p_5s.png': 'tent',
    'p_6s.png': 'table',
    'p_7s.png': 'pine tree',
    'p_8s.png': 'tree',
    'p_9s.png': 'apple tree',
    # sky objects
    's_0s.png': 'helicopter',
    's_1s.png': 'balloon',
    's_2s.png': 'cloud',
    's_3s.png': 'sun',
    's_4s.png': 'cloud with lightning',
    's_5s.png': 'cloud with rain',
    's_6s.png': 'rocket',
    's_7s.png': 'airplane',
    # toys
    't_0s.png': 'baseball',
    't_1s.png': 'basket',
    't_2s.png': 'ball',
    't_3s.png': 'basketball',
    't_4s.png': 'soccer ball',
    't_5s.png': 'tennis ball',
    't_6s.png': 'football',
    't_7s.png': 'frisbee',
    't_8s.png': 'bat',
    't_9s.png': 'air balloons',
    't_10s.png': 'baseball glove',
    't_11s.png': 'shovel',
    't_12s.png': 'racket',
    't_13s.png': 'kite',
    't_14s.png': 'fire',
}
