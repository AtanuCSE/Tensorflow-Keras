# This code is for learning purposes
# It's a complete code to learn Keras at beginner level
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

data = pd.read_csv("fake_reg.csv")  # Loading dataset

#  sns.pairplot(data) #  Data Visualization (Optional)
#  plt.show()

X = data[['feature1', 'feature2']].values  # Creating training attribute array
Y = data['price'].values  # Creating target attribute array


#  Now Split dataset into training and testing using train_test_split function
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 42)

#  Observing shapes (Optional)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

scaler = MinMaxScaler()  # Data pre-processing. Kind of normalization

scaler.fit(X_train)  # Calculating parameters to perform actual scaling. eg. Min, Max, Mean, Std Deviation
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#  model creation technique
#  model = Sequential([Dense(4, activation='relu'), Dense(2, activation='relu'),
#                    Dense(1)])

#  Better model creation technique
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))  # Output layer

model.compile(optimizer='rmsprop', loss='mse')
# mse for continuous data
# binary_crossentropy for binary class
# categorical_crossentropy for multi-class

h = model.fit(x=X_train, y=Y_train, epochs=250, verbose=0)  # Training the model
# verbose = 0 means print no data during training
# verbose > 0 means print information of each step

loss_df = pd.DataFrame(h.history['loss'])
# h.history['loss'] returns loss value of each epoch in an array
# Its for understanding how errors are minimized slowly through each epoch/iteration
# loss_df.plot()  # Visualization
# plt.show()


# Test Data
loss_value_after_evaluate = model.evaluate(X_test, Y_test, verbose=0)
print(loss_value_after_evaluate)
# It returns the loss of the data model

# To get the prediction
test_prediction = model.predict(X_test)
# print(test_prediction)
test_prediction = pd.Series(test_prediction.reshape(Y_test.shape, ))

pred_df = pd.DataFrame(Y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_prediction], axis=1)
pred_df = pred_df.rename(columns={0: 'Model Predictions'})
# print(pred_df)  # Compare

# Plot
# sns.scatterplot(x="Test True Y", y="Model Predictions", data=pred_df)
# plt.show()

# Error Calculation
mae = mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])
mse = mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])
print(mae)
print(mse)

# New data prediction
new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
new_results = model.predict(new_gem)
print(new_results)

# Saving a trained model
model.save('my_fake_reg_trained_model.h5')

# Read the trained model and use
later_model = load_model('my_fake_reg_trained_model.h5')
new_results = later_model.predict(new_gem)
print(new_results)





