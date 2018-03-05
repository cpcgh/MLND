import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively


def get_decision_tree_accuracy(mss):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = mss)
    print "min_samples_split: ", mss

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

acc_min_samples_split_2 = get_decision_tree_accuracy(2)
acc_min_samples_split_50 = get_decision_tree_accuracy(50)


def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}




############### OUTPUT ################
# min_samples_split:  2
# DecisionTree training duration:  0.00145697593689
# DecisionTree predction duration:  0.00019097328186
# min_samples_split:  50
# DecisionTree training duration:  0.00105786323547
# DecisionTree predction duration:  0.00016713142395
# {"message": "{'acc_min_samples_split_50': 0.912, 'acc_min_samples_split_2': 0.908}"}
