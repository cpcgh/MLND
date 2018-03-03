#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB

### create classifier
clf = GaussianNB()

### fit the classifier on the training features and labels
start_time = time()
clf.fit(features_train, labels_train)
print "Training duration: ", time() - start_time

### use the trained classifier to predict labels for the test features
# pred = clf.predict(features_test)

start_time = time()
accuracy = clf.score(features_test, labels_test)
print "Prediction duration: ", time() - start_time
print "accuracy: ", accuracy

#########################################################


