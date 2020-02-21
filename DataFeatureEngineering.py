# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
# House datset from Kaggle is used
# This code shows data exploration, decision based on dataexploration
# Correlation calculation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
