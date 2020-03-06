# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
# Exercise
# Lending Club dataset Kaggle. Can't upload full dataset in github because of it's size

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# load data
data_source = pd.read_csv("lending_club_loan_two.csv")


pd.options.display.width = None  # To see all columns in console while print()

# print(data_source.info())
# See some data are missing. All rows are not same.

# Lets first check balance of the target feature
# loan_status for this dataset
# sns.countplot(data_source['loan_status'])
# plt.show()

# Loan amount histogram
# sns.distplot(data_source['loan_amnt'], bins=25, kde=False, rug=False);
# plt.show()

# Correlation between attributes
# This shows correlation between only numerical value
# corr() can't find correlation for categorical values
# Heat Map clearly indicates the places showing strong correlations
# print(data_source.corr())
# sns.heatmap(data=data_source.corr(), cmap='Greens')
# plt.show()

# Installment is highly correlated with loan_amount. Lets see the scatterplot
# sns.scatterplot(x='installment', y='loan_amnt', data=data_source)
# plt.show()
# This shows that there is a formula between installment and loan_amount
# If loan amount is high then installment will be high
# So, kind of duplicate feature

# Lets check loan status and loan amount relationship
# Loan status is categorical data. So boxplot is required
# sns.boxenplot(x='loan_status', y='loan_amnt', data=data_source)
# plt.show()
# If boxplot is a little hard to understand then we can
# examine the numbers directly between two column
# print(data_source.groupby('loan_status')['loan_amnt'].describe())

# check unique values
# print(data_source['grade'].unique())
# print(data_source['sub_grade'].unique())

# Lets see relationship between grade and loan status
# sns.countplot(x='grade', data=data_source, hue='loan_status')
# plt.show()

# Lets see number of data in each subgrade
# plt.figure(figsize=(10,8))
# sub_grade_order = sorted(data_source['sub_grade'].unique())  # Sort sub_grade types
# sns.countplot(x='sub_grade', data=data_source, order=sub_grade_order, hue='loan_status')  # desired order
# plt.show()
# We can see for F and G there are maximum defaulter
# Thus, lets see only F and G plots
# f_and_g = data_source[(data_source['grade'] == 'G') | (data_source['grade'] == 'F')]
# plt.figure(figsize=(10,8))
# sub_grade_order = sorted(f_and_g['sub_grade'].unique())  # Sort sub_grade types
# sns.countplot(x='sub_grade', data=f_and_g, order=sub_grade_order, hue='loan_status')  # desired order
# plt.show()


# Now we want to change loan_status categorical values to 0 and 1
# But in a new column
data_source['loan_repaid'] = data_source['loan_status'].map({'Fully Paid':1, 'Charged Off':0})


# Lets see correlation between attributes and loan_status come loan_repaid
# data_source.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
# plt.show()

# sns.heatmap(data=data_source.corr(), cmap='Greens')
# plt.show()

# find missing data
# To do that we first need to find null values in each column
print(data_source.isnull().sum())
# In percentage
print(data_source.isnull().sum() / len(data_source) * 100)
# No point of keeping emp_title employment title
data_source = data_source.drop('emp_title', axis=1)
data_source = data_source.drop('emp_length', axis=1)
data_source = data_source.drop('title', axis=1)

# Challenge is to fill mortgage accounts
# print(data_source['mort_acc'].value_counts())
# Which other feature correlates highly with this mortgage account
print(data_source.corr()['mort_acc'])
# looks like total_acc has some effect to some extent
# find mean of mort_acc for each value of total_acc
total_acc_avg = data_source.groupby('total_acc').mean()['mort_acc']
print(total_acc_avg)


# lets make a function to fill data in mort_acc
def mort_acc_fill(total_acc, mort_acc):

    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


data_source['mort_acc'] = data_source.apply(lambda x: mort_acc_fill(x['total_acc'], x['mort_acc']), axis=1)

print(data_source.isnull().sum())

# remove / drop missing rows
data_source = data_source.dropna()
print(data_source.isnull().sum())

# Dealing with categorical data / string data
# Print categorical data first
print(data_source.select_dtypes(['object']).columns)
# Now decide whether to keep them or remove them
print(data_source['term'].value_counts())
# Lets keep the integer part
data_source['term'] = data_source['term'].apply(lambda term: int(term[:3]))

data_source = data_source.drop('grade', axis=1)

# Create dummy variable for subgrade
dummies = pd.get_dummies(data_source['sub_grade'], drop_first=True)
print(dummies)
data_source = pd.concat([data_source.drop('sub_grade', axis=1), dummies], axis=1 )

# Now we will create dummy variable columns for multiple columns
dummies = pd.get_dummies(data_source[['verification_status','application_type','initial_list_status','purpose']], drop_first=True)
data_source = pd.concat([data_source.drop(['verification_status','application_type','initial_list_status','purpose'], axis=1), dummies], axis=1 )

# We can change string values to another string
data_source['home_ownership'] = data_source['home_ownership'].replace(['NONE','ANY'], 'OTHER')
dummies = pd.get_dummies(data_source['home_ownership'], drop_first=True)
data_source = pd.concat([data_source.drop('home_ownership', axis=1), dummies], axis=1 )

# Handle address. Keep zipcode and convert to dummy
data_source['zip_code'] = data_source['address'].apply(lambda address: int(address[-5:]))
dummies = pd.get_dummies(data_source['zip_code'], drop_first=True)
data_source = pd.concat([data_source.drop('zip_code', axis=1), dummies], axis=1 )
data_source = data_source.drop('address', axis=1)


data_source = data_source.drop('issue_d',axis=1)
data_source['earliest_cr_line'] = data_source['earliest_cr_line'].apply(lambda date: int(date[-4:]))



print(data_source)


# End of Data exploration
"""
***********************************************************************
"""


# Loan repaid is 0 and 1, which is actually loan_status
data_source = data_source.drop('loan_status', axis=1)
X = data_source.drop('loan_repaid', axis=1).values
Y = data_source['loan_repaid'].values

#  Now Split dataset into training and testing using train_test_split function
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

# Normalization
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating Layer for new overfitting protection
model = Sequential()
model.add(Dense(78, activation='relu'))  # Good idea to take neurons = number of attribute
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# Binary classification, thus sigmoid
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy')

# Small batch_size will increase time but reduce overfitting chance
h = model.fit(x=X_train, y=Y_train,
              validation_data=(X_test, Y_test),
              epochs=10, batch_size=256)  # Training the model

# Check losses
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