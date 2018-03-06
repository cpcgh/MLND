#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)


### it's all yours from here forward!
features_train = features
labels_train = labels
features_test = features
labels_test = labels

def get_decision_tree_accuracy():
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
    print "DecisionTree predction duration: ", time() - start_time

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)

    return acc

acc = get_decision_tree_accuracy()
print "Accuracy of this decision tree: ", acc


# DecisionTree training duration:  0.000641822814941
# DecisionTree predction duration:  0.000155925750732
# Accuracy of this decision tree:  0.989473684211



from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
acc = get_decision_tree_accuracy()
print "Accuracy of this decision tree: ", acc

# DecisionTree training duration:  0.000371932983398
# DecisionTree predction duration:  7.70092010498e-05
# Accuracy of this decision tree:  0.724137931034
