import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time

def gradient_descent(X,y,features,iterations,learning_rate):
    n = len(X)
    features_length = len(features)
    weights = {}

    for i in range(features_length):
        weights[f'w{i}'] = 0
    
    for i in range(iterations):
        w = np.array(list(weights.values()), dtype=np.float128)

        y_hat = np.dot(X, w.T)

        error = y - y_hat

        loss = np.float128(np.sqrt(((1/(2*n))*np.sum(np.square(error)))))

        if np.all(np.abs(loss) <= 1e-06):
            break

        for j in range(features_length):
            w_deriv = -learning_rate*((1/n)*(np.sum(np.dot(X.iloc[:,j],error))))            
            weights[f'w{j}'] = float(weights[f'w{j}'] - w_deriv)

        print("loss/cost {}, iteration {}".format(loss,i))

    print('weights \n{}'.format(weights))


df = pd.read_csv('Data1.csv')

features = ['T', 'P', 'TC', 'SV']
X = df[features]
y = df['Idx']

# MIN MAX SCALER
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

X[features] = min_max_scaler.fit_transform(X)

# split the data between training and testing
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.5)

for order in range(1,11):
    X_poly_training_data = X_training_data.copy()
    X_poly_testing_data = X_testing_data.copy()

    for power in range(2, order + 1):
        for feature in features:
            X_poly_training_data[f'{feature}^{power}'] = X_poly_training_data[feature]**power
            X_poly_testing_data[f'{feature}^{power}'] = X_poly_testing_data[feature]**power

X_poly_training_data.insert(0, 'intercept', 1)
X_poly_features = list(X_poly_training_data.columns)

gradient_descent(X_poly_training_data,y_training_data,X_poly_features,1000,0.2)
