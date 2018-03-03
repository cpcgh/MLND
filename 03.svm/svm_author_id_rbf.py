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

### Test for speed up SVM
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn.svm import SVC

def train_and_predict(clf):
    print "############# Kernel=", clf.kernel, ", C=", clf.C

    start_time = time()
    clf.fit(features_train, labels_train)
    print "SVM training duration: ", time() - start_time


    start_time = time()
    pred = clf.predict(features_test)
    print "SVM predction duration: ", time() - start_time


    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    print "Accuracy: ", acc

    print "10: ", pred[10]   #1
    print "26: ", pred[26]   #0
    print "50: ", pred[50]   #1

    # Calculate how many emails are from Chris
    chris_count = 0
    for item in pred:
        if item == 1:
            chris_count += 1

    print "Total size of test set: ", len(pred)
    print "Chris emails count: ", chris_count
    print "Sara emails count: ", len(pred) - chris_count




# train_and_predict(SVC(C=10, kernel="rbf"))
# train_and_predict(SVC(C=100, kernel="rbf"))
# train_and_predict(SVC(C=1000, kernel="rbf"))
train_and_predict(SVC(C=10000, kernel="rbf"))


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
###############################
### rbf option
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# SVM training duration:  0.0985610485077
# SVM predction duration:  1.07018494606
# Accuracy:  0.616040955631
### Set value for C
############# Kernel= rbf , C= 10
# SVM training duration:  0.106158971786
# SVM predction duration:  1.02343702316
# Accuracy:  0.616040955631
# ############# Kernel= rbf , C= 100
# SVM training duration:  0.0940940380096
# SVM predction duration:  1.00620794296
# Accuracy:  0.616040955631
# ############# Kernel= rbf , C= 1000
# SVM training duration:  0.0919308662415
# SVM predction duration:  1.00487494469
# Accuracy:  0.821387940842
# ############# Kernel= rbf , C= 10000
# SVM training duration:  0.0939629077911
# SVM predction duration:  0.810650110245
# Accuracy:  0.892491467577
### Use full training set
############# Kernel= rbf , C= 10000
# SVM training duration:  108.991676092
# SVM predction duration:  10.6796779633
# Accuracy:  0.990898748578
