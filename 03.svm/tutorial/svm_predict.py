import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from time import time


import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
start_time = time()
clf.fit(features_train, labels_train)
print "SVM training duration: ", time() - start_time


#### store your predictions in a list named pred
start_time = time()
pred = clf.predict(features_test)
print "SVM predction duration: ", time() - start_time




from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc