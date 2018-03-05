import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from time import time

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here

from sklearn import tree
clf = tree.DecisionTreeClassifier()

start_time = time()
clf.fit(features_train, labels_train)
print "DecisionTree training duration: ", time() - start_time


#### store your predictions in a list named pred
start_time = time()
pred = clf.predict(features_test)
print "DecisionTree predction duration: ", time() - start_time

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

### be sure to compute the accuracy on the test set



def submitAccuracies():
  return {"acc":round(acc,3)}




#################### OUTPUT #################
# DecisionTree training duration:  0.00137996673584
# DecisionTree predction duration:  0.000186204910278
# {"message": "{'acc': 0.908}"}
