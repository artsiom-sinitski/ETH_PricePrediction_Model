#!/usr/bin/env python3
"""
Train and test our prediction model.

Note:
Correct batch-size for LSTM is one
that you can use to divde a number of
samples for both training and testing set by.

So, we have 200 training samples and 50
testing ones, we can use batch-size=1
because we can 200/1 and 50/1, we can
also use 2, can't use 3 because we can't
200/3 and 50/3 etc.
"""
# Configure to get the same
# results every time.

import os, sys
import math
import numpy as np

from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM

from prepData import get_data, prep_data
from matplotlib import pyplot


def get_lstm(batches, input_shape):
    """
    Define and return stateful LSTM.

    Stateful simply means that on every epoch we're not
    starting from scratch, but we're using "remembered"
    sequences from previous epochs which in practice
    means that we should learn "better" and faster.

    input_shape=(# of past data to look for, # of metrics)

    When stateful is True we need to provide batch_input_shape.
    """
    model = Sequential()
    model.add(LSTM(60, input_shape=input_shape, stateful=True, batch_input_Shape=(batches, input_shape[0], input_shape[1])))
    model.add(Dense(1))
    return model


confs = {'default': dict(model=get_lstm)}


def train_model(name, train_x, train_y, epochs, batches, test_x, test_y):
    """
    Get model if it exists, train if needed.
    """