import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
import time


def gradient_descent(x_training, y_training, x_testing, y_testing, features, iterations, learning_rate, order, current_time):
    rmse_training = rmse_testing = r_squared_training = r_squared_testing = tss_training = tss_testing = 0

    weights_training = weights_testing = {}

    features_length = len(features)
    n = len(x_training)

    least_square_training_rmse = 0.12088

    for i in range(features_length):
        weights_training[f'w{i}'] = 0
        weights_testing[f'w{i}'] = 0

    for i in range(iterations):
        w_training = np.array(
            list(weights_training.values()), dtype=np.float128)
        w_testing = np.array(list(weights_testing.values()), dtype=np.float128)

        training_time = "{:.2f}".format(time.time() - current_time)

        # calculate y_hat
        y_hat_training = np.dot(x_training, w_training.T)
        y_hat_testing = np.dot(x_testing, w_testing.T)

        # calculate y_bar
        y_bar_training = np.mean(y_hat_training)
        y_bar_testing = np.mean(y_hat_testing)

        # calculate error
        error_training = y_training - y_hat_training
        error_testing = y_testing - y_hat_testing

        # calculate tss
        tss_training = np.sum(np.square(y_training-y_bar_training))
        tss_testing = np.sum(np.square(y_testing-y_bar_testing))

        # calculate rss
        rss_training = np.sum(np.square(error_training))
        rss_testing = np.sum(np.square(error_testing))

        # calculate mse
        mse_training = ((1/(2*n))*rss_training)
        mse_testing = ((1/(2*n))*rss_testing)

        # calculate rmse
        newRMSE_training = np.sqrt(mse_training)
        newRMSE_testing = np.sqrt(mse_testing)

        '''
            check if the current rmse is close to 0 or close to the 
            rmse calculated by least square method
        '''
        if (i != 0 and (np.abs((newRMSE_training-least_square_training_rmse)) < 0.005 or rmse_training < 0.0005)):
            break
        else:
            rmse_training = newRMSE_training
            r_squared_training = 1 - (rss_training/tss_training)

            rmse_testing = newRMSE_testing
            r_squared_testing = 1 - (rss_testing/tss_testing)

        for j in range(features_length):
            w_deriv_training = -learning_rate * \
                ((1/n)*(np.sum(np.dot(x_training.iloc[:, j], error_training))))
            w_deriv_testing = -learning_rate * \
                ((1/n)*(np.sum(np.dot(x_testing.iloc[:, j], error_testing))))

            weights_training[f'w{j}'] = float(
                weights_training[f'w{j}'] - w_deriv_training)
            weights_testing[f'w{j}'] = float(
                weights_testing[f'w{j}'] - w_deriv_testing)

    print("Order {}, Training Time {} seconds, Testing RMSE {}, Training RMSE {}, Testing R^2 {}, Training R^2 {}, Terms {}".format(
        order, training_time, rmse_testing, rmse_training, r_squared_testing, r_squared_training, features_length-1))


df = pd.read_csv('Data1.csv')

features = ['T', 'P', 'TC', 'SV']
X = df[features]
y = df['Idx']

# MIN MAX SCALER
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

X[features] = min_max_scaler.fit_transform(X)

# split the data between training and testing
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(
    X, y, test_size=0.5)

for order in range(1, 11):
    current_time = time.time()

    X_poly_training_data = X_training_data.copy()
    X_poly_testing_data = X_testing_data.copy()

    for power in range(2, order + 1):
        for feature in features:
            X_poly_training_data[f'{feature}^{power}'] = X_poly_training_data[feature]**power
            X_poly_testing_data[f'{feature}^{power}'] = X_poly_testing_data[feature]**power
    X_poly_training_data.insert(0, 'intercept', 1)
    X_poly_testing_data.insert(0, 'intercept', 1)
    X_poly_features = list(X_poly_training_data.columns)
    gradient_descent(X_poly_training_data, y_training_data, X_poly_testing_data,
                     y_testing_data, X_poly_features, 2000, 0.15, order, current_time)
