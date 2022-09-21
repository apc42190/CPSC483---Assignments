import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
import time

def mean_squared_error(x, y, w0, w1, w2, w3, w4):
    n = len(y)
    total_error = 0
    for i in range(n):
        # get row i of features
        xn1,xn2,xn3,xn4 = x.iloc[i].T.to_numpy()

        # get actual y value for row i
        y_actual = y.iloc[i]

        # calculate y_hat
        y_hat = w0 + w1*xn1 + w2*xn2 + w3*xn3 + w4*xn4

        # calculate error
        calculated_error = y_actual - y_hat

        # square error
        calculated_error**2

        # add to total error
        total_error = total_error + calculated_error

    # return MSE
    return (1/(2*n)) * total_error

def gradient_descent(x,y,iterations,learning_rate):
    w0 = w1 = w2 = w3 = w4 = 0

    for i in range(iterations):
        # calculate current loss/cost value
        loss = mean_squared_error(x,y,w0,w1,w2,w3,w4)


        print('here\n')

        # calculate the partial derivative for each weight coefficient
        xn1,xn2,xn3,xn4 = x.iloc[i].T.to_numpy()
        w0d = -learning_rate*loss
        w1d = -learning_rate*loss*xn1
        w2d = -learning_rate*loss*xn2
        w3d = -learning_rate*loss*xn3
        w4d = -learning_rate*loss*xn4

        w0 = w0 - w0d
        w1 = w1 - w1d
        w2 = w2 - w2d
        w2 = w2 - w2d
        w3 = w3 - w3d
        w4 = w4 - w4d

        print("w0 {}, w1 {}, w2 {}, w3 {}, w4 {}, loss/cost {}, iteration {}".format(w0,w1,w2,w3,w4,loss,i))

df = pd.read_csv('Data1.csv')

features = ['T', 'P', 'TC', 'SV']
X = df[features]
y = df['Idx']

# split the data between training and testing
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.4)

gradient_descent(X_training_data,y_training_data,10,0.00001)
