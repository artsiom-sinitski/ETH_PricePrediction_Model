"""
Prepare ETH prices data for use in an LSTM network

The goal is to predict a closing porice of ETH token on a given day based on the metrics from the previous months.

We're working with historical ETH prices data in .CSV format
from yahoo finances portal -> https://finance.yahoo.com.

Each row is a set of metrics for a given day.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot


def get_data(file='data/ETH-USD_prices.csv'):
    """
    Load Ethereum token prices data from a .csv file,
    and convert it to the correct form.
    """
    data = pd.read_csv(file)
    x, y = get_raw_xy(data)
    yy = get_vpo(y)

    return x[:-1], yy[:-1]


def get_raw_xy(data):
    """
    First return value: values that will be used for predictions except for 'Date' & 'Adj Close' columns.
    Second return value: a list of 'Close Price' column.
    """
    # Removing 'Date' & 'Adj Close' columns
    # Assuming that our data don't have any dividents, so close column is the same as adj close
    data = data.drop(columns=['Date', 'Adj Close'])
    values = data.values
    # Each column number matches a specific metric:
    # Open=0, High=1, Low=2, Close=3, Volume=4
    return values[:, [0,1,2,3,4]], values[:, 3]


def get_vpo(values):
    """
    This function shifts values one index backwards.

    Day_1: m11, m12, *m13
    Day_2: m21, m22, *m23
    Day_3: m31, m32, *m33

    We want to predict values with *, so

    If we want to train our network to predict Day_1
    we don't have any data from the previous day, so we can't
    do that, we base prediction of m23 on metrics from prev data:
    [m11, m12, m13] and we can do the same for Day_2:

    X: m11,m12,m13 Y: m23
    X: m21,m22,m23 Y: m33

    What about data from Day_3? Well, we don't have any
    data from Day_4 to use for prediction using metrics
    from Day_3.

    So, this is how we're constructing our data:
    X: No data      Y: m13     <- We discard this first value,
    X: m11,m12,m13  Y: m23        since we don't have any X data from Day_0
    X: m21,m22,m23  Y: m33
    X: m31,m32,m32  Y: No data <- We need to discard this as well, since                                 there's no data for Y from Day_4
    """
    shifted_y = list(values)
    shifted_y.pop(0)
    shifted_y.append(None)
    return shifted_y


def prep_data(train_x, train_y):
    """
    Split data and return in the exact format
    we need it for our LSTM to learn
    """
    train_x, train_y, test_x, test_y = split_train_test(train_x, train_y)

    # We need one more dimension (we need to put our values
    # into one more list) for our x values
    train_x = np.array(np.expand_dims(train_x, axis=1))
    train_y = np.array(train_y)
    
    test_x = np.array(np.expand_dims(test_x, axis=1))
    test_y = np.array(test_y)

    print("Sets shapes: ", train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


def split_train_test(X, Y, trs_len=0.80):
    """
    Split both X and Y into train and test sets.

    trs_len - how much data should we use for training?
              by default it's 0.80 meaning 80%, the remining
              20% of the data will be used for testing.
    """
    lx = len(X)
    trs = int(lx * trs_len)
    train_x, train_y = X[:trs], Y[:trs]
    test_x, test_y = X[trs:], Y[trs:]
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # Here we want to first show how the "shifting" works
    # then graph our training and test data.
    data = pd.read_csv('data/ETH-USD_prices.csv')
    x, y = get_raw_xy(data)
    yy = get_vpo(y)

    print("\n----- Data before conversion -----")
    for i in range(5):
        print('X[%d]/Y[%d]\n' % (i, i), x[i], ' ==> ', y[i])
    
    print("\n----- Shifted -----")
    for i in range(5):
        print('X[%d]/YY[%d]\n' % (i, i), x[i], ' ==> ', yy[i])

    train_x, train_y, test_x, test_y = split_train_test(x, yy)

    # Prepare data for plotting.
    p_tx = list(train_x[:, 3]) + [None]*(len(test_x))
    p_ttx = ([None]*(len(train_x)-1)) + list(train_x[:, 3])[-1:] + list(test_x[:, 3])

    # Plot closing prices for each day.
    pyplot.plot(p_tx, label='train_x')
    pyplot.plot(p_ttx, label='test_x')
    pyplot.legend()
    pyplot.show()

    x, y = get_data()

    print("\n----- Data before preparation -----")
    print(x[0], ' ==> ', y[0])
    print("\n----- Data after preparation -----")
    train_x, train_y, test_x, test_y = prep_data(x, y)
    print(train_x[0], ' ==> ', train_y[0])
