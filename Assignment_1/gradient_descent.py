import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time

def gradient_descent(X,y,iterations,learning_rate):
    w0 = w1 = w2 = w3 = w4 = 0
    n = len(X)

    for i in range(iterations):
        w = np.array([w1,w2,w3,w4])
        y_hat = X.dot(w.T)
        loss = (1/(2*n))*np.sum(np.square(y_hat-y))

        if np.all(np.abs(loss) <= 1e-06):
            break

        w0d = learning_rate*(1/n)*np.sum((y_hat - y))
        w1d = learning_rate*(1/n)*np.sum(X.iloc[:, 0] * (y_hat - y))
        w2d = learning_rate*(1/n)*np.sum(X.iloc[:, 1] * (y_hat - y))
        w3d = learning_rate*(1/n)*np.sum(X.iloc[:, 2] * (y_hat - y))
        w4d = learning_rate*(1/n)*np.sum(X.iloc[:, 3] * (y_hat - y))
        
        w0 = w0 - w0d
        w1 = w1 - w1d
        w2 = w2 - w2d
        w3 = w3 - w3d
        w4 = w4 - w4d

        # TODO: create a tolerance to skip loop

        print("w0 {}, w1 {}, w2 {}, w3 {}, w4 {}, loss/cost {}, iteration {}".format(w0,w1,w2,w3,w4,loss,i))

df = pd.read_csv('Data1.csv')

features = ['T', 'P', 'TC', 'SV']
X = df[features]
y = df['Idx']

# split the data between training and testing
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.4)

gradient_descent(X_training_data,y_training_data,1000,0.00001)
