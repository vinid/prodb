#!/usr/bin/env python

"""Tests for `prodb` package."""


import unittest
from click.testing import CliRunner
from dataclasses import dataclass
import keras
import tensorflow as tf
from prodb import prodb

@dataclass
class Config:
    MAX_LEN = 20
    BATCH_SIZE = 256
    LR = 0.001
    VOCAB_SIZE = 33489
    EMBED_DIM = 64
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 64  # used in bert model
    NUM_LAYERS = 1
    LOSS_FN = keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    LOSS_TRACKER = tf.keras.metrics.Mean(name="loss")


config = Config()
