#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the default configuration of the experiment, with values that can
optionally be passed as command line args.
"""

import argparse

LABEL_MAP = {'not_icr': 0, 'icr': 1}


def args() -> argparse.Namespace:
    """Parse arguments given by user and return the name space."""
    parser = argparse.ArgumentParser(
        description='Training the iCR recognition models in CoDraw task.')
    # _________________________________ PATHS _________________________________
    parser.add_argument('-path_to_codraw',
                        default='./data/CoDraw-master/dataset/CoDraw_1_0.json',
                        type=str, help='Path to CoDraw data JSON file.')
    parser.add_argument('-path_to_annotation',
                        default='./data/cr_anno_adjudication/data_adju.tsv',
                        type=str, help='Path to iCR annotation tsv file.')
    parser.add_argument('-annotation_column_name',
                        default='is CR? v2',
                        type=str, help='Which annotator to use (column name).'\
                        'in the annotation tsv file.')
    parser.add_argument('-path_to_preprocessed_imgs',
                        default='./data/preprocessed/images/',
                        type=str, help='Path to dir of image embeddings.')
    parser.add_argument('-path_to_preprocessed_texts',
                        default='./data/preprocessed/texts/',
                        type=str, help='Path to dir of text embeddings.')
    parser.add_argument('-path_to_checkpoints',
                        default='./checkpoints/',
                        type=str, help='Path to dir where to log checkpoints.')
    parser.add_argument('-path_to_outputs',
                        default='./outputs/',
                        type=str, help='Path to dir where to log checkpoints.')
    # _________________________________ COMET _________________________________
    parser.add_argument('-ignore_comet', action='store_true',
                        help='Do not log details to Comet_ml.')
    parser.add_argument('-comet_key', default='',
                        type=str, help='Comet.ml personal key.')
    parser.add_argument('-comet_project', default='icr-codraw-eacl2023',
                        type=str, help='Comet.ml project name.')
    parser.add_argument('-comet_workspace', default='',
                        type=str, help='Comet.ml workspace name.')
    # ______________________________ SETTING __________________________________
    parser.add_argument('-random_seed', default=35466, type=int,
                        help='Random seed set for reproducibility.')
    parser.add_argument('-device', default='gpu', type=str,
                        choices=['cpu', 'gpu'], help='Which device to use.')
    parser.add_argument('-task', default='drawer', type=str,
                        choices=['teller', 'drawer'],
                        help='Which task to train classifier on.')
    parser.add_argument('-text_pretrained', default='all-mpnet-base-v2',
                        type=str, help='Which pretrained text encoder to use.')
    parser.add_argument('-img_pretrained', default='resnet101', type=str,
                        help='Which pretrained image encoder to use.')
    parser.add_argument('-no_context', action='store_true',
                        help='Do not use dialogue context as input.')
    parser.add_argument('-no_image', action='store_true',
                        help='Do not use image as input.')
    parser.add_argument('-no_msg', action='store_true',
                        help='Do not use the last utterance as input.')
    parser.add_argument('-downsample', default=1, type=float,
                        help='Proportion of not-iCR to include in train set.')
    parser.add_argument('-upsample', default=0, type=int,
                        help='Number of duplicates for each iCR in train set.')
    parser.add_argument('-only_with_icrs', action='store_true',
                        help='Do not use dialogues with no iCRs.')
    parser.add_argument('-only_until_peek', action='store_true',
                        help='Use dialogues only up to the peek action.')
    parser.add_argument('-remove_first', action='store_true',
                        help='Ignore first turns.')
    parser.add_argument('-delay_scenes', action='store_true',
                        help='Predictions should use the state of the scene' \
                             'at the last turn, before current actions begin.')

    # __________________________ TRAINING PARAMS ______________________________
    parser.add_argument('-batch_size', default=128, type=int,
                        help='Batch size.')
    parser.add_argument('-img_input_dim', default=2048, type=int,
                        help='Size of pretrained image embedding.')
    parser.add_argument('-context_input_dim', default=768, type=int,
                        help='Size of pretrained context embedding.')
    parser.add_argument('-last_msg_input_dim', default=768, type=int,
                        help='Size of pretrained utterance embedding.')
    parser.add_argument('-img_embedding_dim', default=128, type=int,
                        help='Size of internal image embedding.')
    parser.add_argument('-context_embedding_dim', default=128, type=int,
                        help='Size of internal context embedding.')
    parser.add_argument('-last_msg_embedding_dim', default=128, type=int,
                        help='Size of internal utterance embedding.')
    parser.add_argument('-hidden_dim', default=256, type=int,
                        help='Classifier hidden layer dimension.')
    parser.add_argument('-n_epochs', default=20, type=int,
                        help='Number of epochs to train the model.')
    parser.add_argument('-dropout', default=0.1, type=float,
                        help='Droupout.')
    parser.add_argument('-weight_cr', default=2.6125454767515217, type=float,
                        help='Weight for positive class in loss function,' \
                             '0 for automatic computation.')
    parser.add_argument('-decision_threshold', default=0.5, type=float,
                        help='Threshold for regression decision.')
    parser.add_argument('-clip', default=1, type=float,
                        help='Clipping size, use 0 for no clipping.')
    parser.add_argument('-accumulate_grad', default=25, type=int,
                        help='Steps for batch gradient accumulation.')
    parser.add_argument('-weight_decay', default=0.0001, type=float,
                        help='Weight decay for L2 regularisation.')
    parser.add_argument('-lr', default=0.003, type=float,
                        help='Learning rate.')
    parser.add_argument('-lr_scheduler', default='exp', type=str,
                        choices=['none', 'exp', 'step'],
                        help='Which lr scheduler to use.')
    parser.add_argument('-lr_step', default=2, type=int,
                        help='Which step to use if lr_scheduler is step.')
    parser.add_argument('-gamma', default=0.99, type=float,
                        help='Gamma for the LR scheduler.')

    return parser.parse_args()
