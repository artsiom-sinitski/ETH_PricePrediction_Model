"""
Predict closing price of the Ethereum token.

Important:
Batch size that you choose will determine the
number of predictions you can make.
ex:
batch_size=1 = 1 prediction
batch_size=5 = 5 predictions, etc...

Have a look at trainModel.py for informations about
what the value of batch_size should be.
"""
import os

from keras.models import Sequential, load_model
from keras.layers import Dense

from trainModel import confs, get_params
import configRes
import numpy as np

# Below I am using real ETH price data from 07-22-2018.
# [ [Open, High, Low, Close(t), Volume], Close(t+1) ]
to_predict = [
               [461.04, 471.99, 455.01, 457.65, 102716195], 449.63
             # [457.65, 469.70,	446.61,	449.63, 175179390], 479.47
             ]

# this is the same as input_shape to our LSTM model.
# (num. of past days of data to use, num. of features to use)
data_shape = (1, 5)


if __name__ == '__main__':
    # Get command line params
    name, epochs, batches, _ = get_params(script='predictPrice.py')
    model = confs[name]
    mname = 'models/model-%s-%d-%d.h5 ' % (name, epochs, batches)

    # Loading the model
    if os.path.exists(mname):
        model = load_model(mname)
        print('\n***** Model <%s> loaded! *****\n' % name)
    else:
        print("Can't find %s model, train it first using 'trainModel.py %s %d %d'" % (mname, name, epochs, batches))

    # conacatenate feature arrays for Day_10 & Day_11
    p = np.array(to_predict[0])
    # Convert data into the "right" format.
    p = np.reshape(p, (batches, data_shape[0], data_shape[1]))
    # Get the expected price for validation.
    #c = to_predict[0][1] "c == Close(t+1) price"
    c = to_predict[1]
    # Again here we need to specify the batch_size.
    x = model.predict(p, batch_size=batches, verbose=1)
    # We have just one prediction.
    x = x[0][0]

    print('Predicted: $%.2f' % x)
    print('   Actual: $%.2f' % c)
    print('    Error: $%.2f (%.2f%%)' % (x-c, abs((x-c)*100/c)) )
