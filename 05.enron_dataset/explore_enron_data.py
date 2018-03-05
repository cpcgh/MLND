#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print "Size of dataset: ", len(enron_data)

poi_count = 0
quantified_salary_employees_count = 0
known_email_address_count = 0

for person in enron_data:
    print "Name: ", person
    print "Number of features: ", len(enron_data[person])
    print "Features: ", enron_data[person]
    if 1 == enron_data[person]['poi']:
        poi_count += 1

    if enron_data[person]['salary'] != "NaN":
        quantified_salary_employees_count += 1

    if enron_data[person]['email_address'] != "NaN":
        known_email_address_count += 1

print "Number of POI: ", poi_count
print "Stock value of James Prentice: ", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "POI emails to Wesley Colwell: ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Exercised stocks of SKILLING JEFFREY K: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]


print "total_payments of FASTOW ANDREW S: ", enron_data["FASTOW ANDREW S"]["total_payments"]
print "total_payments of SKILLING JEFFREY K: ", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "total_payments of LAY KENNETH L: ", enron_data["LAY KENNETH L"]["total_payments"]

print "quantified_salary_employees_count: ", quantified_salary_employees_count
print "known_email_address_count: ", known_email_address_count
