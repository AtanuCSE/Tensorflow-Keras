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

# Visualize data to understand the dataset
# Distribute the labels
# plt.figure(figsize=(10, 6))
# sns.distplot(df['price'])
# plt.show()
# sns.countplot(df['bedrooms'])
# plt.show()

# Check correlation between labels / attributes
# Target is price in this example
# Higher value means higher correlation with the price
print(df.corr()['price'].sort_values())
# Plot correlation between price and most high correlated attribute sqft_living
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()
# Another visualization idea
plt.figure(figsize=(10, 6))
sns.boxenplot(x='bedrooms', y='price', data=df)
plt.show()
# More data exploration using logtitude and lattitude of Kind County
# This plot show in which area belongs to High Price Flats
plt.figure(figsize=(10, 8))
sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.show()
# This plot is informative but inclues 10-20 extremely priced houses
# Thus, leaving those houses from prediction is a good idea

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]
plt.figure(figsize=(10, 8))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None,
                alpha=0.5, palette='RdYlGn', hue='price')
plt.show()

sns.boxenplot(x='waterfront', y='price', data=df)
plt.show()
