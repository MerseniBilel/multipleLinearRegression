from os import system
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

system("clear")
## file name : FuelConsumption.csv
### read file
df=pd.read_csv("FuelConsumption.csv")
## print(df.head())
## print(df.columns)
## print(df.describe())

## clean csv file
x = df[['ENGINESIZE','CYLINDERS']]
y = df['CO2EMISSIONS']

msk = np.random.rand(len(df)) < 0.9
train = df[msk]
test = df[~msk]


train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
print(regr.coef_) 
predictedCO2 = regr.predict([[2.4, 4]])
print(predictedCO2)
