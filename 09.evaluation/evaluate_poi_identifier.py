#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from time import time
import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)
print "features: ", features
print "labels: ", labels
# overfit model
features_train = features
labels_train = labels
features_test = features
labels_test = labels



###### define functions ########
def get_decision_tree_pred_and_accuracy():
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    # print "min_samples_split: ", mss

    print "Number of features: ", len(features_train[0])
    start_time = time()
    clf.fit(features_train, labels_train)
    print "DecisionTree training duration: ", time() - start_time


    #### store your predictions in a list named pred
    start_time = time()
    pred = clf.predict(features_test)
    print "DecisionTree prediction duration: ", time() - start_time

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)

    return pred, acc

def convert_list_to_dict(key_list, value_list):
    dict = {}
    for i in range(len(key_list)):
        key = np.asscalar(key_list[i])
        value = np.asscalar(value_list[i])
        dict[key] = value
    print "convert_list_to_dict: ", dict
    return dict
###################################


pred, acc = get_decision_tree_pred_and_accuracy()
print "Accuracy of this decision tree: ", acc


def calc_true_positive_count(features_test, labels_test, pred):
    over_fit_data = convert_list_to_dict(features_test, pred)
    test_data = convert_list_to_dict(features_test, labels_test)
    intersects = [k for k in over_fit_data if k in test_data]
    # print "Intersects:", filter(over_fit_data.has_key, test_data.keys())
    print "\nIntersects:", intersects

    true_positive_count = 0
    for k in intersects:
        if over_fit_data[k] == test_data[k] and over_fit_data[k] == 1:
            print "True Positive: ", k
            true_positive_count += 1

    print "\ntrue_positive_count: ", true_positive_count

calc_true_positive_count(features_test, labels_test, pred)
# True Positive:  274975.0
# True Positive:  224305.0
# True Positive:  240189.0
# True Positive:  243293.0
# True Positive:  365163.0
# True Positive:  309946.0
# True Positive:  1072321.0
# True Positive:  158403.0
# True Positive:  1111258.0
# True Positive:  420636.0
# True Positive:  288542.0
# True Positive:  249201.0
# True Positive:  440698.0
# True Positive:  211844.0
# True Positive:  415189.0
# True Positive:  213999.0
#
# true_positive_count:  16


# Cross validation model
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

print "features_test: ", features_test
print "labels_test: ", labels_test

test_set_poi_count = 0
for is_poi in labels_test:
    if is_poi == 1:
        test_set_poi_count += 1

print "test_set_poi_count: ", test_set_poi_count
print "total people in test set count: ", len(labels_test)


pred, acc = get_decision_tree_pred_and_accuracy()
print "Accuracy of this decision tree: ", acc
# DecisionTree training duration:  0.000371932983398
# DecisionTree predction duration:  7.70092010498e-05
# Accuracy of this decision tree:  0.724137931034


calc_true_positive_count(features_test, labels_test, pred)
# true_positive_count:  0


# Calculate the precision score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "precision_score: ", precision_score(labels_test, pred)
print "recall_score: ", recall_score(labels_test, pred)
# precision_score:  0.0
# recall_score:  0.0

####### QUIZ
pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
labels_test = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print "precision_score: ", precision_score(labels_test, pred)
print "recall_score: ", recall_score(labels_test, pred)
# precision_score:  0.666666666667
# recall_score:  0.75
