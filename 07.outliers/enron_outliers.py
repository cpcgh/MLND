#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

# Find out who had higest salary or bonus
for name in data_dict:
    salary = data_dict[name]['salary']
    bonus = data_dict[name]['bonus']
    if salary != "NaN" and salary > 1000000 and bonus != "NaN" and bonus > 5000000:
        print "name: ", name, ", salary: ", salary , ", bonus: ", bonus

    # if bonus != "NaN" and bonus > 5000000:
    #     print "name: ", name, ", salary: ", salary, ", bonus: ", bonus


# Remove the outliers i.e. TOTAL
data_dict.pop( "TOTAL", 0 )

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
