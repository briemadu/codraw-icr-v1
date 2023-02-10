#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract utterance embeddings and incremental, truncated dialogue context
embeddings using SentenceTransformers.
https://www.sbert.net/docs/pretrained_models.html

Following their usage, as describede here: https://www.sbert.net/

MODEL_NAME, EMB_DIM and MAX_SEQ_LEN can be passed via the CLI. 
Select one SentenceTransformer pretrained model, pass the embedding 
dimension accordingly and pick a number lower than the model's maximum
sequence length to truncate the context (we don't pick the maximum itself
because of subtokenization).

For each split, this script builds 2 h5py objects. All of them contain one
dataset for each scene ID in CoDraw. The names/keys are the game IDs.

1. 'codraw_utterances_{SPLIT}.hdf5'
2. 'codraw_dialogues_{SPLIT}.hdf5'

In 1, each dataset contains a tensor of dim (n_rounds, 2, EMB_DIM),
and the two dimensions in the middle correspond to (teller, drawer).
In 2, each dataset contains a tensor of dim (n_rounds, 2, EMB_DIM),
and the two dimensions in the middle correspond to (context before teller,
context after teller but before drawer).
"""
import argparse
import os
import sys
sys.path.append('.')
from pathlib import Path
import h5py
import json
import numpy as np
from numpy import concatenate as cat
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from icr.structs.game import Game

argParser = argparse.ArgumentParser()
argParser.add_argument("-model", type=str, default='all-mpnet-base-v2',
                       help="Pretrained model name")
argParser.add_argument("-dimension", type=int, default=768,
                       help="Pretrained model output dim")
argParser.add_argument("-max_seq_len", type=int, default=200,
                       help="Maximum input sequence len, bounded by model.")
argParser.add_argument("-remove_teller", action='store_true',
                       help="Exclude teller's turns from contexts.")
argParser.add_argument("-remove_drawer", action='store_true',
                       help="Exclude drawer's turns from contexts.")
args = argParser.parse_args()

MODEL_NAME = args.model
EMB_DIM = args.dimension
MAX_SEQ_LEN = args.max_seq_len
print(f'Generating image embeddings using {MODEL_NAME}.')
print(f'remove_teller is set to {args.remove_teller}.')
print(f'remove_drawer is set to {args.remove_drawer}.')

SPLITS = ('train', 'val', 'test')
DATA = './data/'
CODRAW_PATH = Path(f'{DATA}CoDraw-master/dataset/CoDraw_1_0.json')
DEST = f'{DATA}preprocessed/texts/'
try:
    os.mkdir(DEST)
except FileExistsError:
    pass
ablation = ''
if args.remove_teller:
    ablation += '_no-teller'
if args.remove_drawer:
    ablation += '_no-drawer'
OUTPUT_DIR = Path(f'{DEST}{MODEL_NAME}{ablation}/')
os.mkdir(OUTPUT_DIR)

EMBS = 'codraw_utterances_{}.hdf5'
CUM_EMBS = 'codraw_dialogues_{}.hdf5'

# Define which separators to use between turns.
SEP_TELLER = '/T '
SEP_DRAWER = '/D '
SEP_PEEK = '/PEEK'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL_NAME).to(DEVICE)


with open(CODRAW_PATH, 'r') as file:
    codraw = json.load(file)
# collect all game IDs in each split
idxs = {split: [] for split in SPLITS}
for name, game in codraw['data'].items():
    split = name.split('_')[0]
    idxs[split].append(name)


def truncate(text):
    """Truncate left context, otherwise model probably truncates right."""
    return " ".join(text.split()[-MAX_SEQ_LEN:])


def retrieve_embeddings(encoder, turn_seq, remove_teller, remove_drawer):
    """Get the sequence of incremental embeddings for one full dialogue."""
    # initialize empty context
    context = ''
    # store the sequence of (teller's emb, drawer's emb)
    seq = []
    # store the sequence of dialogue contexts embeddings
    # (before round, after teller's utterance)
    # the first "before round" one is always an empty context
    # the one after the last drawer's utterance is not
    # necessary because it is not a context for any observed
    # next round
    cum_seq = []

    for t, turn in enumerate(turn_seq):
        # context before round, i.e. empty at first and then
        # after drawer's last utterance
        embedding_c1 = encoder.encode([context])
        # teller turn embedding
        embedding_t = encoder.encode([turn.teller])
        # context after teller and before drawer
        if not remove_teller:
            # add a marker before the round if a peek occurs
            if t == game.peek_turn:
                context += f' {SEP_PEEK}'
                context = truncate(context)
            # if remove_teller was true, we'd encode a copy of the previous
            # context instead
            context += f' {SEP_TELLER}{turn.teller}'
            context = truncate(context)
        embedding_c2 = encoder.encode([context])
        # drawer turn embedding
        embedding_d = encoder.encode([turn.drawer])
        # update context with drawer turn, for next context
        if not remove_drawer:
            # if remove_drawer was true, we'd encoder a copy of the previous
            # context instead
            context += f' {SEP_DRAWER}{turn.drawer}'
            context = truncate(context)
        # the last context won't be necessary because there is
        # no utterance after the drawer's last utterance

        # create tuples for this round
        seq_round = cat([embedding_t, embedding_d], axis=0)
        cum_seq_round = cat([embedding_c1, embedding_c2], axis=0)
        # update sequences
        seq.append(seq_round)
        cum_seq.append(cum_seq_round)

    return seq, cum_seq


with torch.no_grad():
    for split in SPLITS:
        embeddings = OUTPUT_DIR / EMBS.format(split)
        cumulative_embeddings = OUTPUT_DIR / CUM_EMBS.format(split)
        with h5py.File(embeddings, 'w') as out_embs:
            with h5py.File(cumulative_embeddings, 'w') as out_cum_embs:
                for name in tqdm(idxs[split]):
                    # passing an empty iCR set, because not needed here
                    game = Game(name, codraw['data'][name], set())
                    dialogue = game.dialogue.turns
                    game_id = str(int(name.split('_')[1]))
                    n_rounds = len(codraw['data'][name]['dialog'])
                    assert n_rounds == game.dialogue.n_turns
                    # call the sequence of forward passes in the encoders
                    sequence, cum_sequence = retrieve_embeddings(
                        model, dialogue,
                        remove_teller=args.remove_teller,
                        remove_drawer=args.remove_drawer)
                    sequence = np.array(sequence)
                    cum_sequence = np.array(cum_sequence)
                    assert sequence.shape == (n_rounds, 2, EMB_DIM)
                    assert cum_sequence.shape == (n_rounds, 2, EMB_DIM)
                    out_embs.create_dataset(game_id, data=sequence)
                    out_cum_embs.create_dataset(game_id, data=cum_sequence)
