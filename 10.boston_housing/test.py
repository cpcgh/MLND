
# Import libraries necessary for this project
import numpy as np
import pandas as pd
# from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
# import visuals as vs

# Pretty display for notebooks
# %matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
print "Type of prices:", type(prices)
print "Prices:\n", prices
print np.min(prices)
print np.max(prices)
print np.mean(prices)
print np.median(prices)
print np.std(prices)
