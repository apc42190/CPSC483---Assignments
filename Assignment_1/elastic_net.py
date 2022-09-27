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
df = pd.read_csv('Data1.csv')


#Divide dataframe into depenant and independent variables
features = ['T', 'P', 'TC', 'SV']
X = df[features].copy()
y = df['Idx']


current_time = time.time()

# MIN MAX SCALER
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scaled feature
X[features] = min_max_scaler.fit_transform(X)

# Split data between training and testing
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.5)


order = 4
# Generate Higher Order Terms: No Interations
for power in range(2, order + 1):
    for feature in features:
        X_training_data[f'{feature}^{power}'] = X_training_data[feature] ** power
        X_testing_data[f'{feature}^{power}'] = X_testing_data[feature] ** power

for alpha in [x * 0.001 for x in range(1, 10)]:


    # Train model
    model = linear_model.ElasticNet(alpha = alpha, l1_ratio=0.9)
    model.fit(X_training_data, y_training_data)

    training_time = "{:.2f}".format(time.time() - current_time)


    #Test model with training data
    y_hat = model.predict(X_training_data)
    training_RMSE = "{:.5f}".format(mean_squared_error(y_training_data, y_hat, squared = False)) 
    training_R_2 = "{:.5f}".format(r2_score(y_training_data, y_hat))

    #Test model with testing data
    y_hat = model.predict(X_testing_data)
    testing_RMSE = "{:.5f}".format(mean_squared_error(y_testing_data, y_hat, squared = False)) 
    testing_R_2 = "{:.5f}".format(r2_score(y_testing_data, y_hat))

    l =  "{:.3f}".format(alpha)

    print(f'Lambda: {l}, Training Time: {training_time} secs, Testing RMSE: {testing_RMSE}, Training RMSE: {training_RMSE}, Testing R^2: {testing_R_2}, Training R^2: {training_R_2}')
    # print(f'Intercept: {model.intercept_}, Weights: {model.coef_}\n')
