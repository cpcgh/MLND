#!/usr/bin/python
import pickle

enron_data = pickle.load(open("../final_project/poi_names.txt", "r"))

# TODO: Fix this
print "Size of dataset: ", len(enron_data)
