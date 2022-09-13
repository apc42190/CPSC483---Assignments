import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time


df = pd.read_csv('Data1.csv')

X = df[['T', 'P', 'TC', 'SV']]
y = df['Idx']

X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 1)

for order in range(1, 10):
    polinomial_features = PolynomialFeatures(degree = order)

    current_time = time.time()
    X_poly_training_data = polinomial_features.fit_transform(X_training_data)
    model = linear_model.LinearRegression()
    model.fit(X_poly_training_data, y_training_data)
    training_time = time.time() - current_time

    y_hat = model.predict(X_poly_training_data)

    print(f'Order = {order}, RMSE = {mean_squared_error(y_training_data, y_hat, squared = False)}, R^2 = {r2_score(y_training_data, y_hat)}, training_time = {training_time} seconds')
