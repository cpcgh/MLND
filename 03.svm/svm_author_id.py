#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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

# ### Test for speed up SVM
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="linear")


start_time = time()
clf.fit(features_train, labels_train)
print "SVM training duration: ", time() - start_time


start_time = time()
pred = clf.predict(features_test)
print "SVM predction duration: ", time() - start_time


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "Accuracy: ", acc
#########################################################



########### Output ############
### Original option
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# SVM training duration:  165.044928074
# SVM predction duration:  16.3883259296
# Accuracy:  0.984072810011
###############################
### 1/100 option
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# SVM training duration:  0.0939679145813
# SVM predction duration:  1.03559398651
# Accuracy:  0.884527872582
