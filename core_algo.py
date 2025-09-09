import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from six.moves import urllib
# from IPython.display import clear
import tensorflow.compat.v2.feature_column as fc

# linear Regression

"""
x = [1, 2, 2.5, 3, 5]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

"""

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(dftrain.loc[0])
# print("=>",y_train.loc[0])


print(y_train.head())
dftrain.age.hist(bins=20)
plt.show()

# 1:27:28