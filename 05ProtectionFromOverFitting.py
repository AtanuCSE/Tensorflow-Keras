# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
# Protection from overfitting problem
# Confusion Matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# load data
df = pd.read_csv("cancer_classification.csv")

# Check the balance of target label
# sns.countplot(df['benign_0__mal_1'])
# plt.show()

# Lets check Correlation with target
# print(df.corr()['benign_0__mal_1'])

# Another correlation check - heatmap
# sns.heatmap(df.corr())
# plt.show()

# Train and Test set
# Separate Label from attributes / features
X = df.drop('benign_0__mal_1', axis=1).values  # values is for panda to numpy conversion
Y = df['benign_0__mal_1'].values

# Split Train Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # No Fit for Test Set

# Creating Layer
model = Sequential()
model.add(Dense(30, activation='relu'))  # Good idea to take neurons = number of attribute
model.add(Dense(15, activation='relu'))

# Binary classification, thus sigmoid
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# Small batch_size will increase time but reduce overfitting chance
h = model.fit(x=X_train, y=Y_train,
              validation_data=(X_test, Y_test),
              batch_size=128, epochs=600)  # Training the model

# See that loss and validation loss goes apart
# That means overfitting
loss_df = pd.DataFrame(h.history['loss'])
ax = loss_df.plot()
loss_df2 = pd.DataFrame(h.history['val_loss'])
loss_df2.plot(ax=ax)
plt.show()

# Overfitting can be handled with early stopping
# min is because we want to minimize loss
# We will use max when want to maximize something
# for example: we want to maximize accuracy
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# Patience means that the model doesn't stop immidiately after val_loss changes
# It should see 25(random) steps to be sure that overfitting is started occuring

# Creating Layer
model = Sequential()
model.add(Dense(30, activation='relu'))  # Good idea to take neurons = number of attribute
model.add(Dense(15, activation='relu'))

# Binary classification, thus sigmoid
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# Small batch_size will increase time but reduce overfitting chance
h = model.fit(x=X_train, y=Y_train,
              validation_data=(X_test, Y_test),
              batch_size=128, epochs=600, callbacks=[early_stopping])  # Training the model

# Check early stopping effects
loss_df = pd.DataFrame(h.history['loss'])
ax = loss_df.plot()
loss_df2 = pd.DataFrame(h.history['val_loss'])
loss_df2.plot(ax=ax)
plt.show()


# Creating Layer for new overfitting protection
model = Sequential()
model.add(Dense(30, activation='relu'))  # Good idea to take neurons = number of attribute
model.add(Dropout(0.5))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

# Binary classification, thus sigmoid
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# Small batch_size will increase time but reduce overfitting chance
h = model.fit(x=X_train, y=Y_train,
              validation_data=(X_test, Y_test),
              epochs=600, callbacks=[early_stopping])  # Training the model

# Check early stopping effects
loss_df = pd.DataFrame(h.history['loss'])
ax = loss_df.plot()
loss_df2 = pd.DataFrame(h.history['val_loss'])
loss_df2.plot(ax=ax)
plt.show()

# Test the set
predictions = model.predict_classes(X_test)

# Precision, Recall, F1-Score, Confusion Matrix
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))

