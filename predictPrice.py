"""
Predict closing price of the Ethereum token.

Important:
Batch size that you choose will determine the
number of predictions you can make.
ex:
batch_size=1 = 1 prediction
batch_size=5 = 5 predictions, etc...

Have a look at train.py for informations about
what the value of batch_size should be.
"""
import os

from keras.models import Sequential, load_model
from keras.layers import Dense

from trainModel import confs, get_params
import configRes
import numpy as np


# [ [Open, High, Low, Close(t), Volume], Close(t+1) ]
to_predict = [[296.94,298.19,288.00,288.95,8350500], 280.74]
# this is the same as input_shape to our LSTM models
# (num of past days of data to use, num of metrics to use)
data_shape = (1, 5)


if __name__ == '__main__':
    # Get command line params.
    name, epochs, batches, _ = get_params(script='predictPrice.py')
    model = confs[name]
    mname = 'models/model-%s-%d-%d.h5' % (name, epochs, batches)
    # Loading the model.
    if os.path.exists(mname):
        model = load_model(mname)
        print('Model loaded!')
    else:
        print("Can't find %s model, train it first using 'train.py %s %d %d'" % (mname, name, epochs, batches))
    p = np.array(to_predict[0])
    # Convert data into the "right format".
    p = np.reshape(p, (batches, data_shape[0], data_shape[1]))
    # Get the expected price for validation.
    c = to_predict[0][1]
    # Again here we need to specify the batch_size.
    x = model.predict(p, batch_size=batches)
    # We have just one prediction.
    x = x[0][0]
    print('Predicted $%.2f, actual $%.2f, error $%.2f (%.2f%%)' % (x, c, x-c, abs((x-c)*100/c)))
