import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time


#Load data into Pandas dataframe
df = pd.read_csv('Data1.csv')

#Divide dataframe into depenant and independent variables
features = ['T', 'P', 'TC', 'SV']
X = df[features]
y = df['Idx']

#Split data between training and testing
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.4)

for order in range(1, 11):
    
    current_time = time.time()
    
    #Generate Higher Order Terms: Include Interations
    # polynomial_features = PolynomialFeatures(degree = order)
    # X_poly_training_data = polynomial_features.fit_transform(X_training_data)
    # X_poly_testing_data = polynomial_features.fit_transform(X_testing_data)
    
    #Generate Higher Order Terms: No Interations
    X_poly_training_data =  X_training_data.copy()
    X_poly_testing_data = X_testing_data.copy()
    for power in range(2, order + 1):
        for feature in features:
            X_poly_training_data[f'{feature}^{power}'] = X_poly_training_data[feature]**power
            X_poly_testing_data[f'{feature}^{power}'] = X_poly_testing_data[feature]**power
    
    #Train model
    model = linear_model.LinearRegression()
    model.fit(X_poly_training_data, y_training_data)
    

    training_time = "{:.2f}".format(time.time() - current_time)


    #Test model with testing data
    y_hat = model.predict(X_poly_testing_data)
    testing_RMSE = "{:.5f}".format(mean_squared_error(y_testing_data, y_hat, squared = False)) 
    testing_R_2 = "{:.5f}".format(r2_score(y_testing_data, y_hat))

    print(f'Order: {order}, Training Time: {training_time} secs, RMSE: {testing_RMSE}, R^2: {testing_R_2}')
    #print(f'Intercept: {model.intercept_}, Weights: {model.coef_}\n')
