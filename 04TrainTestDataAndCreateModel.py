# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
# House datset from Kaggle is used

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, explained_variance_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
df = pd.read_csv('kc_house_data.csv')

# Check if any missing data
# null_value = df.isnull().sum()
# print(null_value)

# id attribute has no effectiveness in this dataset
df = df.drop('id', axis=1)

# Change date string to date time object
# This will help to utilize month, year etc.
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
print(df.head())
# lambda expression means kind of creating a inline function call
# def year_extraction(date):
#   return date.year
# This extraction of new information is called feature engineering
# print(df.groupby('month').mean()['price'])
# This shows that no significance between price and month

df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)
# zipcode is here a numeric value.
# This will hamper prediction coz it doesn't have any significance
# If we want to find significance that will require more domain knowledge
# Such as which zipcode areas have more priced flats

print(df['sqft_basement'].value_counts())
# This will give an idea if this column is necessary
# After observing the count we can decide if it's relevant with price

# Separate Label from attributes / features
X = df.drop('price', axis=1).values  # values is for panda to numpy conversion
Y = df['price'].values

# Split Train Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # No Fit for Test Set

# Creating Layer
model = Sequential()
model.add(Dense(19, activation='relu'))  # Good idea to take neurons = number of attribute
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Small batch_size will increase time but reduce overfitting chance
h = model.fit(x=X_train, y=Y_train,
              validation_data=(X_test, Y_test),
              batch_size=128, epochs=200)  # Training the model

loss_df = pd.DataFrame(h.history['loss'])
loss_df.plot()
plt.show()

# Predictions
predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(Y_test, predictions)))
print(mean_absolute_error(Y_test, predictions))
# The errors are high due to the fact that a very small
# number of data has a high value, which has effect on the common values
# while training
# Lets check this using visualization

plt.figure(figsize=(12, 6))
plt.scatter(Y_test, predictions)
plt.show()
# See few number of plots more that 3000000
# But those value has effect on training
# An attempt to correct this could be leaving those 1% values from dataset
# and build a model for the common values only


# Testing on a new data
# Creating a new data from training set
single_house = df.drop('price', axis=1).iloc[0]  # Numpy array

single_house = scaler.transform(single_house.values.reshape(-1, 19))
print(model.predict(single_house))





