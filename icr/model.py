#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the neural network model architecture.
"""

import argparse

import torch
from torch import nn, Tensor

# only one output neuron for binary task modeled with regression
OUTPUT_DIM = 1


class Classifier(nn.Module):
    """Model used for iCR classification tasks."""
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self._extract_config(config)
        concat_dim = self._define_concat_dim(config)
        if self.use_image:
            self.img_encoder = ImageEncoder(
                config.img_input_dim, config.img_embedding_dim)
        if self.use_msg:
            self.msg_encoder = TextEncoder(
                config.last_msg_input_dim, config.last_msg_embedding_dim)
        if self.use_context:
            self.context_encoder = TextEncoder(
                config.context_input_dim, config.context_embedding_dim)
        self.classifier = DeeperClassifier(
            concat_dim, self.output_dim, config.dropout, config.hidden_dim)

    def _extract_config(self, config: argparse.Namespace) -> None:
        self.use_context = not config.no_context
        self.use_image = not config.no_image
        self.use_msg = not config.no_msg
        self.output_dim = OUTPUT_DIM
        assert self.use_context or self.use_image or self.use_msg

    def _define_concat_dim(self, config: argparse.Namespace) -> int:
        internal_dim = 0
        if self.use_msg:
            internal_dim += config.last_msg_embedding_dim
        if self.use_image:
            internal_dim += config.img_embedding_dim
        if self.use_context:
            internal_dim += config.context_embedding_dim
        assert internal_dim > 0
        return internal_dim

    def forward(self, context: Tensor, last_msg: Tensor,
                image: Tensor) -> Tensor:
        """Perform a forward pass and return the logits."""
        # initialize as an empty tensor as it for sure will be
        # appended to in either image or context
        x_input = torch.tensor([]).to(last_msg.device)
        if self.use_msg:
            encoded_msg = self.msg_encoder(last_msg)
            x_input = torch.cat([x_input, encoded_msg], dim=1)
        if self.use_image:
            encoded_image = self.img_encoder(image)
            x_input = torch.cat([x_input, encoded_image], dim=1)
        if self.use_context:
            encoded_context = self.context_encoder(context)
            x_input = torch.cat([x_input, encoded_context], dim=1)
        output = self.classifier(x_input)
        return output


class DeeperClassifier(nn.Module):
    """Linear classifier with two layers."""
    def __init__(self, concat_dim: int, output_dim: int, dropout: float,
                 hidden_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Retrieve logits for the given input."""
        return self.classifier(x)


class ImageEncoder(nn.Module):
    """Linear classifier for the image."""
    def __init__(self, img_input_dim: int, img_embedding_dim: int):
        super().__init__()
        self.encoder = nn.Linear(img_input_dim, img_embedding_dim)

    def forward(self, image: Tensor) -> Tensor:
        """Retrieve embedding for the image."""
        return self.encoder(image)


class TextEncoder(nn.Module):
    """Linear classifier for the texts."""
    def __init__(self, text_input_dim: int, text_embedding_dim: int):
        super().__init__()
        self.encoder = nn.Linear(text_input_dim, text_embedding_dim)

    def forward(self, text: Tensor) -> Tensor:
        """Retrieve embedding for the input text."""
        return self.encoder(text)
