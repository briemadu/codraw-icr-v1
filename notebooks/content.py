#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A list with content words to be used in the trivial baselines. The words were
selected manually by inspecting a sample of dialogues.
"""

SELECTED_TOKENS = set([
    # CLIPARTS
    # animals: a*.png
    'bear', 'cat', 'dog', 'duck', 'owl', 'bird', 'snake',
    # clothing: c*.png
    'hat', 'cap',  'star', 'crown', 'queen', 'king', 'chef', 'pirate',
    'rainbow', 'witch', 'wizard', 'brim', 'viking', 'horns', 'horn', 'helmet',
    'sunglasses', 'propeller', 'glasses', 'striped',
    # food: e*.png
    'pie', 'pizza', 'hot', 'dog', 'hotdog', 'ketchup', 'mustard', 'hamburger',
    'burger', 'cup', 'drink', 'soda', 'drinking', 'glass', 'bottle', 'straw',
    # people: hb*.png
    'girl', 'girls', 'boy', 'boys', 'mike', 'jenny', 'kid', 'kids', 'child',
    'children', 'dude', 'woman', 'man',
    # large objects: p*.png
    'bee', 'ride', 'playground', 'slide', 'handles', 'stairs', 'sandbox',
    'dune', 'sand', 'box', 'mound', 'hump',
    'grill', 'swing', 'swings', 'seat', 'tent', 'door', 'flap', 'table',
    'picnic', 'tree', 'bushy', 'apple', 'apples', 'oak', 'hole', 'trunk',
    'leafy', 'bush', 'pine', 'bubble', 'fluffy', 'rung', 'leaf', 'leaves',
    'chestnut', 'bumblebee', 'knot',
    # sky objects: s*.png
    'helicopter', 'blades', 'windshield', 'chopper', 'hot', 'air', 'balloon',
    'balloons', 'strings', 'sun', 'rays', 'travel', 'basket', 'cloud',
    'clouds', 'lightning', 'rain', 'rainy', 'bump', 'drops', 'bolt',
    'thundering', 'thunder', 'rocket', 'ship', 'airplane', 'plane',
    # toys
    'baseball', 'bucket', 'pail', 'beach', 'ball', 'basketball', 'soccer',
    'tennis', 'football', 'ballpoint', 'frisbee', 'bat', 'party', 'glove',
    'mitt', 'shovel', 'racket', 'kite', 'string', 'fireplace', 'fire',
    'campfire', 'flame', 'toy',
    # BACKGROUND
    'grass', 'horizon', 'air', 'sky', 'skyline', 'vegetation',
    # EMOTIONS
    'angry', 'happy', 'sad', 'shock', 'shocked', 'surprised', 'stressed',
    'scared', 'mad', 'crying', 'worried',
    # ACTIONS & BODY POSITIONS
    'jumping', 'sitting', 'running', 'cross', 'legged', 'smiling',
    'outstretched', 'wearing', 'pointing', 'yoga', 'stretched', 'looking',
    'standing', 'poking', 'racing', 'waving', 'kicking', 'kicks', 'throwing',
    'indian', 'reaching', 'touching', 'frowning', 'reaching', 'carrying',
    'holding', 'extended', 'walking', 'hiding', 'cheering', 'sliding',
    # SIZES
    'size', 'giant', 'large', 'medium', 'small', 'big', 'smaller', 'bigger',
    'larger', 'tiny',
    # POSITIONS AND SIDES
    'facing', 'left', 'right', 'top', 'bottom', 'above', 'below', 'center',
    'middle', 'underneath', 'front', 'side', 'position', 'corner', 'edge',
    'side', 'level', 'inch', 'between', 'behind', 'posture', 'border',
    'direction', 'up', 'down', 'lower', 'higher', 'upper', 'tip', 'close',
    'horizontal', 'vertical', 'straight', 'depth',
    # BODY PARTS
    'body', 'arm', 'arms', 'feet', 'foot', 'shoulder', 'shoulders', 'head',
    'heads', 'hand', 'hands', 'eye', 'eyes', 'leg', 'legs', 'hair', 'smile',
    'face', 'faces', 'tongue', 'tail', 'paw', 'neck', 'waist', 'teeth',
    'tooth', 'wrist', 'pinky', 'finger', 'fingers', 'pigtail', 'elbow',
    'elbows', 'wing', 'wings', 'knee', 'knees', 'eyebrows', 'armpits',
    'armpit', 'hips', 'nose', 'noses', 'mouth', 'mouths', 'back', 'grimace',
    'toe', 'toes', 'chest', 'chests', 'torso', 'wrist', 'wrists', 'bill',
    'thumb', 'thumbs', 'ponytails', 'ponytail', 'ear', 'ears',
    'chin', 'forehead',
    # COLORS
    'color', 'colour', 'red', 'gray', 'blue', 'yellow', 'brown', 'purple',
    'pink', 'multicolored', 'orange', 'grey', 'brown', 'green', 'black',
    'white',
    # CLOTHES
    'pants', 'shoe', 'shoes', 'shorts', 'shirt', 'sleeves',
    # PRONOUNS
    'he', 'she', 'his', 'him', 'her', 'they', 'them',
    # OTHER
    'where', 'which', 'what', 'type', 'color', 'size', 'how', 'far', 'visible',
])
