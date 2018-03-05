#!/usr/bin/python

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



reg = studentReg(ages_train, net_worths_train)

print "Katie's net worth prediction: ", reg.predict([[27]])
print "Slope: ", reg.coef_
print "Intercept: ", reg.intercept_

print "\n################## stats on test dataset #################\n"
print "r-squared score: ", reg.score(ages_test, net_worths_test)
print "\n################## stats on training dataset #################\n"
print "r-squared score: ", reg.score(ages_train, net_worths_train)

plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())








# Katie's net worth prediction:  [[160.43205453]]
# Slope:  [[6.47354955]]
# Intercept:  [-14.35378331]
#
# ################## stats on test dataset #################
#
# r-squared score:  0.812365729231
#
# ################## stats on training dataset #################
#
# r-squared score:  0.874588235822
