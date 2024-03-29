#!/usr/bin/env python3
"""
Train and test our prediction model.
The data is 1 year period of ETH prices:
from July 14th, 2018 - August 13th, 2019. 

Note:
Correct batch-size for LSTM is one
that you can use to divde a number of
samples for both training and testing set by.

So, if we have 200 training samples and 50
testing ones, we can use batch-size=1
because we can 200/1 and 50/1, we can
also use 2, can't use 3 because we can't
200/3 and 50/3 etc.

To train the model type in the CLI: "python3 trainModel.py <model_name> <# of epochs> <batch size>"
    Ex: $> python3 trainModel.py 'default' 100 2
"""

import os, sys
import math
import numpy as np

from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM
from keras import regularizers as rgl

from prepData import get_data, prep_data
from matplotlib import pyplot


def get_lstm(batches, input_shape):
    """
    Define and return stateful LSTM.

    Stateful simply means that on every epoch we're not
    starting from scratch, but we're using "remembered"
    sequences from previous epochs which in practice
    means that we should learn "better" and faster.

    input_shape = (# of past data to look at, # of features)
    This argument is required if using LSTM as the first layer
    in a model.

    When stateful is True we need to provide batch_input_shape.
    """

    # Will stack layers on top of each other,
    # so initialize the model as sequential.
    model = Sequential()
    model.add( LSTM(64, input_shape=input_shape, stateful=True, batch_input_shape=(batches, input_shape[0], input_shape[1])) ) 
    
    model.add(Dense(1, kernel_regularizer=rgl.l2(0.01),                                        activity_regularizer=rgl.l1(0.01)))
                #kernel_regularizer=regularizers.l2(0.01)
    return model


confs = {'default': dict(model=get_lstm)}


def train_model(name, train_x, train_y, epochs, batches, test_x, test_y):
    """
    Configure and train the model.
    """
    mparams = confs[name]
    model = mparams['model'](batches, (train_x.shape[1], train_x.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mape'])

    # Since the stateful LSTM is learning dependencies between data points 
    # we want to keep the data in the same order on each iteration,
    # thus we don't want to shuffle it.
    history = model.fit(train_x, train_y, verbose=2, epochs=epochs, batch_size=batches, validation_data=(test_x, test_y), shuffle=False)

    return model, name, mparams, history


def get_params(script='trainModel.py'):
    """
    Get command line parameters
    """
    xa = ''
    if script == 'trainModel.py':
        xa = 'ploth'
    try:
        name, epochs, batches = sys.argv[1:4]
        if not hasattr(sys, 'argv'):
            sys.argv  = ['']
    except ValueError:
        print('Usage: %s model_name epochs batch_size %s' % (script, xa))
        exit(1)

    try:
        plot = sys.argv[4]
    except IndexError:
        plot = False
    
    return name, int(epochs), int(batches), plot


if __name__ == '__main__':
    X, Y = get_data()
    train_x, train_y, test_x, test_y = prep_data(X,Y)

    # Getting our command line parameters
    name, epochs, batches, plot = get_params()

    # Do the training
    model, name, mp, history = train_model(name, train_x, train_y, epochs, batches, test_x, test_y)
    print("\n***** The model has been trained! *****\n")

    # Save models and the training history for later use
    mname = 'models/model-%s-%d-%d' % (name, epochs, batches)
    model.save(mname + '.h5')
    title = '%s (epochs=%d, batch_size=%d)' % (name, epochs, batches)
    
    # Test our model on both data that has been seen
    # (training data set) and unseen (test data set)
    print("----- Scores for %s model -----" % title)
    # Notice that we need to specify 'batch_size" in the
    # evaluate() function when we're using an LSTM layer.
    train_score = model.evaluate(train_x, train_y, verbose=0, batch_size=batches)
    trscore = "| RMSE: $%s | MAPE: %.0f%%" % ("{:,.0f}".format(math.sqrt(train_score[0])), train_score[2])
    print("Train Score %s" % trscore)
    test_score = model.evaluate(test_x, test_y, verbose=0, batch_size=batches)
    tscore = "| RMSE: $%s | MAPE: %.0f%%" % ("{:,.0f}".format(math.sqrt(test_score[0])), test_score[2])
    print(' Test Score %s' % tscore)

    # Plot history
    if plot:
        pyplot.plot([ math.sqrt(l) for l in history.history['loss'] ], label='train RMSE ($)')
        pyplot.plot([ math.sqrt(l) for l in history.history['val_loss'] ], label='test RMSE ($)')
        pyplot.legend()
        pyplot.show()

        pyplot.plot(history.history['mean_absolute_percentage_error'], label='train mape')
        pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='test mape')
        pyplot.legend()
        pyplot.show()