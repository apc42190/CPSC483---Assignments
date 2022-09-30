import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time

#Data to be read
#df = pd.read_csv('test-scaled-data.csv')
df = pd.read_csv('../Data1.csv')


#Divide dataframe into depenant and independent variables
features = ['T', 'P', 'TC', 'SV']
X = df[features].copy()
y = df['Idx']


current_time = time.time()

# MIN MAX SCALER
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scaled feature
X[features] = min_max_scaler.fit_transform(X)


order = 4
# Generate Higher Order Terms: No Interations
for power in range(2, order + 1):
    for feature in features:
        X[f'{feature}^{power}'] = X[feature] ** power
        X[f'{feature}^{power}'] = X[feature] ** power


# Train model
model = linear_model.RidgeCV(alphas = [0.0001, 0.001, 0.1], cv = 5)
model.fit(X, y)

training_time = "{:.2f}".format(time.time() - current_time)


#Test model with training data
y_hat = model.predict(X)
RMSE = "{:.5f}".format(mean_squared_error(y, y_hat, squared = False)) 
R_2 = "{:.5f}".format(r2_score(y, y_hat))

print(f'alpha: {model.alpha_}, Training Time: {training_time} secs, RMSE: {RMSE}, R^2: {R_2}')
#print(f'Intercept: {model.intercept_}, Weights: {model.coef_}\n')