"""
Configuration to get the same results every time.

https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
"""
import os
import random as rn

import numpy as np

import tensorflow as tf
from keras import backend as K

# Set up random seed to
# get the same results every
# time you train your model.
rs=5

# We want to silence some of the
# tensorflow log messages for the clarity
# of the output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(rs)
rn.seed(rs)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(rs)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
