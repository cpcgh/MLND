#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    # print "Predictions: ", predictions
    # print "Ages: ", ages
    # print "Networths: ", net_worths

    list = []
    for i, age in enumerate(ages):
        t = (age, net_worths[i], abs(predictions[i] - net_worths[i]))
        list.append(t)

    list = sorted(list, key=lambda t: t[2])
    # print "list: ", list


    cleaned_data = list[0 : int(len(list) * 0.9)]
    print "Size of cleaned data: ", len(cleaned_data)

    ### your code goes here


    return cleaned_data
