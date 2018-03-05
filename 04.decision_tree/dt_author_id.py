#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

def get_decision_tree_accuracy(mss):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = mss)
    print "min_samples_split: ", mss

    print "Number of features: ", len(features_train[0])
    start_time = time()
    clf.fit(features_train, labels_train)
    print "DecisionTree training duration: ", time() - start_time


    #### store your predictions in a list named pred
    start_time = time()
    pred = clf.predict(features_test)
    print "DecisionTree predction duration: ", time() - start_time

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)

    return acc

acc = get_decision_tree_accuracy(40)
print acc

#########################################################



################ OUTPUT ##################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# min_samples_split:  40
# DecisionTree training duration:  69.7493479252
# DecisionTree predction duration:  0.0387599468231
# 0.977246871445

########### Set percentile as 1 instead of 10
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# min_samples_split:  40
# Number of features:  379
# DecisionTree training duration:  4.29435110092
# DecisionTree predction duration:  0.00202488899231
# 0.965870307167
