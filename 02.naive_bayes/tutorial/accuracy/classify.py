def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    
    j = 0
    match = 0
    for item in pred:
        if (item == labels_test[j]):
            # print "item: ", item
            # print "label: ", labels_test[j]
            match+=1
        j+=1
    
    print "accurate items: ", match 
    print "total items: ", len(pred)
    print "accuracy of option 1: ", 1.0 * match / len(pred)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)
    print "accuracy of option 2: ", accuracy
    
    return accuracy