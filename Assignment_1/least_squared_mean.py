import seaborn as sns
import numpy as np
import pandas as pd


df = sns.load_dataset('iris')
print(df.head())

dependent_column = 'sepal_length'
features = ['intercept', 'sepal_width', 'petal_length', 'petal_width']
df['intercept'] = 1


coefficients = []

for order in range(1, 10):
    modified_df = df
    betas = []
    for feature in features:
        column = df[[feature]]**order
        beta = np.linalg.inv(column.T.dot(column)).dot(column.T).dot(df[dependent_column].fillna(0))    
        betas.append(beta[0])
    coefficients.append(betas)

for function in coefficients:
    print(function)