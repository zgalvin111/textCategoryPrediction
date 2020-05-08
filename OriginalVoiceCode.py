import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import LinearRegression
import xlwt
import csv
import timeit

# PARAMETERS
#   n_nonzero - <type 'list'> - Desired number of non-zero entries in the solution
#   X_train - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 3/4 of all the X's. Used to train the dataset
#   y_train - <class 'pandas.core.series.Series'> - 3/4 of all the y's. Used to
# train the dataset
#   X_test - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 1/4 of all the X's. Used to test the trained model
#   y_test - <class 'pandas.core.series.Series'> - 1/4 of all the y's. Used to
# test the trained dataset.
# PURPOSE
#   The purpose of CheckOrthogonalMatchingPursuit is to find the optimzed parameters based upon
# the data that is fed to it and the returned that model trained with all of the
# the data
# RETURNS
#   ompPred - <type 'list'>
#       0. The prediction accuracy of the optimized model
#       1. The optimized parameter(s)
#       2. The total time that the function took
def CheckOrthogonalMatchingPursuit(n_nonzero, X_train, y_train, X_test, \
    y_test):
        # define start time
        startTime = timeit.default_timer()

        # 1. Instatiate OrthogonalMatching Pursuit & create dict of test parameters
        omp = OrthogonalMatchingPursuit()
        params = dict(n_nonzero_coefs=n_nonzero)

        # n_iter can never be great than the alpha range. Thus, for now we will
        # set n_iter equal to the length of alpha_range. This should be changed
        # for larger data sets
        iter = len(n_nonzero)

        # 2. Instatiate the Randomized Search to prepare for alpha testing
        rand = RandomizedSearchCV(omp, params, n_iter=iter, random_state=4)

        # 3. This model uses dense arrays to convert all inputs to dense arrays
        # the y values do not have to be converted because they are pandas series
        X_train = X_train.todense([])
        X_test = X_test.todense([])

        # 4. Fit the training data to the the randomized search object to find
        # the optimal parameters based upon the testing data
        rand.fit(X_train, y_train)

        # 5. Instatiate NEW Mulinomial Naive Bais with optimal parameters
        optimized_omp = OrthogonalMatchingPursuit(n_nonzero_coefs= \
            rand.best_params_['n_nonzero_coefs'])

        # 6. Fit the optimized model with our training data
        optimized_omp.fit(X_train, y_train)

        # 7. Make predictions on test data based on optimized model
        y_pred_class = optimized_omp.predict(X_test)

        # Record ending time
        endTime = timeit.default_timer()
        finalTime = endTime - startTime

        # 8. OrthogonalMatchingPursuit returns continuous predictions and so, in order to
        # find the accuracy of the integers, you must change the type to int
        y_pred_class = y_pred_class.astype(int)

        # 9. Calculate accuracy of the predictions above
        ompPred = [metrics.accuracy_score(y_test, y_pred_class), rand.best_params_['n_nonzero_coefs'], finalTime]

        # 10. Return the accuracy of the optimized model
        return ompPred


# PARAMETERS
#   alpha_range - <type 'list'> - Smoothing Parameter
#   X_train - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 3/4 of all the X's. Used to train the dataset
#   y_train - <class 'pandas.core.series.Series'> - 3/4 of all the y's. Used to
# train the dataset
#   X_test - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 1/4 of all the X's. Used to test the trained model
#   y_test - <class 'pandas.core.series.Series'> - 1/4 of all the y's. Used to
# test the trained dataset.
# PURPOSE
#   The purpose of CheckMultiNomialNB is to find the optimized MultinomialNB model
# based upon the data it is given
# RETURNS
#   mlbPred - <type 'list'>
#       0. The prediction accuracy of the optimized model
#       1. The optimized parameter(s)
#       2. The total time that the function took
def CheckMultiNomialNB(alpha_range, X_train, y_train, X_test, y_test):
        # define start time
        startTime = timeit.default_timer()

        # 1. Instatiate MultinomialNB & create dictionary of test parameters
        mlb = MultinomialNB()
        params = dict(alpha=alpha_range)

        # n_iter can never be great than the alpha range. Thus, for now we will
        # set n_iter equal to the length of alpha_range. This should be changed
        # for larger data sets
        iter = len(alpha_range)

        # 2. Instatiate the Randomized Search to prepare for alpha testing
        rand = RandomizedSearchCV(mlb, params, n_iter=iter, random_state=4)

        # 3. Fit the training data to the the randomized search object to find
        # the oprimal parameters based upon the testing data
        rand.fit(X_train, y_train)

        # 4. Instatiate NEW Mulinomial Naive Bais with optimal parameters
        nb = MultinomialNB(alpha=rand.best_params_['alpha'])

        # 5. Fit the optimized model with our training data
        nb.fit(X_train, y_train)

        # 6. Make predictions on test data based on optimized model
        y_pred_class = nb.predict(X_test)

        # Record ending time
        endTime = timeit.default_timer()
        finalTime = endTime - startTime

        # 7. Calculate accuracy of the predictions above
        mlbPred = [metrics.accuracy_score(y_test, y_pred_class), rand.best_params_['alpha'], finalTime]

        # 8. Return the accuracy of the optimized model
        return mlbPred

# PARAMETERS
#   l_alpha - <type 'list'> - Constant that multiplies the L1 term. Defaults to
# 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the
# LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso
# object is not advised. Given this, you should use the LinearRegression object.
#   X_train - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 3/4 of all the X's. Used to train the dataset
#   y_train - <class 'pandas.core.series.Series'> - 3/4 of all the y's. Used to
# train the dataset
#   X_test - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 1/4 of all the X's. Used to test the trained model
#   y_test - <class 'pandas.core.series.Series'> - 1/4 of all the y's. Used to
# test the trained dataset.
# PURPOSE
#   The purpose of CheckLasso is to find the optimized Lasso Regression model
# based upon the data that is provided.
# RETURNS
#   lasPred - <type 'list'>
#       0. The prediction accuracy of the optimized model
#       1. The optimized parameter(s)
#       2. The total time that the function took
def CheckLasso(l_alpha, X_train, y_train, X_test, y_test):
    # define start time
    startTime = timeit.default_timer()

    # 1. Instatiate Lasso Regression & create dictionary of test parameters
    l = Lasso()
    params = dict(alpha=l_alpha)

    # n_iter can never be great than the alpha range. Thus, for now we will
    # set n_iter equal to the length of alpha_range. This should be changed
    # for larger data sets
    iter = len(l_alpha)

    # 2. Instatiate the Randomized Search to prepare for alpha testing
    rand = RandomizedSearchCV(l, params, n_iter=iter, random_state=4)

    # 3. Fit the training data to the the randomized search object to find
    # the oprimal parameters based upon the testing data
    rand.fit(X_train, y_train)

    # 4. Instatiate NEW Lasso Regression with optimal parameters
    las = Lasso(alpha=rand.best_params_['alpha'])

    # 5. Fit the optimized model with our training data
    las.fit(X_train, y_train)

    # 6. Make predictions on test data based on optimized model
    y_pred_class = las.predict(X_test)

    # Record ending time
    endTime = timeit.default_timer()
    finalTime = endTime - startTime

    # 7. Lasso regression returns continuous predictions and so, in order to
    # find the accuracy of the integers, you must change the type to int
    y_pred_class = y_pred_class.astype(int)

    # 8. Calculate accuracy of the predictions above
    lasPred = [metrics.accuracy_score(y_test, y_pred_class), rand.best_params_['alpha'], finalTime]

    # 9. Return the accuracy of the optimized model
    return lasPred

# PARAMETERS
#   neighbors - <type 'list'> - Number of neighbors to use by default for
# kneighbors queries.
# object is not advised. Given this, you should use the LinearRegression object.
#   X_train - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 3/4 of all the X's. Used to train the dataset
#   y_train - <class 'pandas.core.series.Series'> - 3/4 of all the y's. Used to
# train the dataset
#   X_test - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 1/4 of all the X's. Used to test the trained model
#   y_test - <class 'pandas.core.series.Series'> - 1/4 of all the y's. Used to
# test the trained dataset.
# PURPOSE
#   The purpose of CheckKNearest is to find the optimized Lasso Regression model
# based upon the data that is provided.
# RETURNS
#   lasPred - <type 'list'>
#       0. The prediction accuracy of the optimized model
#       1. The optimized parameter(s)
#       2. The total time that the function took
def CheckKNearest(neighbors, X_train, y_train, X_test, y_test):
    # define start time
    startTime = timeit.default_timer()

    # 1. Instatiate K Nearest Neighbors & create dictionary of test parameters
    k = KNeighborsClassifier()
    params = dict(n_neighbors=neighbors)

    # n_iter can never be great than the alpha range. Thus, for now we will
    # set n_iter equal to the length of alpha_range. This should be changed
    # for larger data sets
    iter = len(neighbors)

    # 2. Instatiate the Randomized Search to prepare for alpha testing
    rand = RandomizedSearchCV(k, params, n_iter=iter, random_state=4)

    # 3. Fit the training data to the the randomized search object to find
    # the oprimal parameters based upon the testing data
    rand.fit(X_train, y_train)

    # 4. Instatiate NEW K Nearest Neighbors with optimal parameters
    kn = KNeighborsClassifier(n_neighbors=rand.best_params_['n_neighbors'])

    # 5. Fit the optimized model with our training data
    kn.fit(X_train, y_train)

    # 6. Make predictions on test data based on optimized model
    y_pred_class = kn.predict(X_test)

    # Record ending time
    endTime = timeit.default_timer()
    finalTime = endTime - startTime

    # 7. Calculate accuracy of the predictions above
    kPred = [metrics.accuracy_score(y_test, y_pred_class), rand.best_params_['n_neighbors'], finalTime]

    # 8. Return the accuracy of the optimized model
    return kPred
# PARAMETERS
#   X_train - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 3/4 of all the X's. Used to train the dataset
#   y_train - <class 'pandas.core.series.Series'> - 3/4 of all the y's. Used to
# train the dataset
#   X_test - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is
# 1/4 of all the X's. Used to test the trained model
#   y_test - <class 'pandas.core.series.Series'> - 1/4 of all the y's. Used to
# test the trained dataset.
#   X_all - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that is used
# to fit the best optimized model before it is returned
#   y - <class 'pandas.core.series.Series'> - All of the y's that is used to fit
# the best optimized model before it is returned
#   mlbRange (optional) - <type 'list'> - Alpha range for the MultinomialNB. It
# is used as a smoothing parameter.
#   lassoRange (optional) - <type 'list'> - Constant that multiplies the L1 term. Defaults to
# 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the
# LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso
# object is not advised. Given this, you should use the LinearRegression object.
#   kRange (optional) - <type 'list'> - Number of neighbors to use by default for
# kneighbors queries.
#   OMPRange (optional) - <type 'list'> - Desired number of non-zero entries in
# the solution
# PURPOSE
#   The purpose of findBestOptimizedModel is to go through as many models as
# possible, optimize them, test them and then compare the test scores. The model
# with the best test score or the model that acheives 100% accuracy first will
# best chosen as the best optimized model.
# RETURNS
#   Model - <class 'sklearn.bestModel'> - Where bestModel can be any one of the
# models that are tested. The model that is returned will be fitted with all of
# X and Y as well.
def findBestOptimizedModel(X_train, y_train, X_test, y_test, X_all, y,
    mlbRange = range(0,10),lassoRange = range(-10,10), kRange = range(1,6),
    OMPRange = range(1,10)):

    # 1. Test each model to find the model with the best prediction accuracy
    mlbPred = CheckMultiNomialNB(mlbRange, X_train, y_train, X_test,
        y_test)

    # Print Results
    print("\n")
    print("MultinomailNB Accuracy: " + str(mlbPred[0]) + " in " + str(mlbPred[2]) \
        + " seconds")
    print("\n")

    lasPred = CheckLasso(range(-10,10), X_train, y_train, X_test,
        y_test)

    # Print Results for Lasso Regression
    print("\n")
    print("Lasso Regression Accuracy: " + str(lasPred[0]) + " in " + \
        str(lasPred[2]) + " seconds")
    print("\n")

    kPred = CheckKNearest(range(1,6), X_train, y_train, X_test, y_test)

    # Print results for K Nearest neighbors
    print("\n")
    print("K Nearest Kneighbors Accuracy: " + str(kPred[0]) + " in " + \
        str(kPred[2]) + " seconds")
    print("\n")

    ompPred = CheckOrthogonalMatchingPursuit(range(1,10), X_train, y_train, \
        X_test, y_test)

    # Print results for Orthogonal Matching Pursuit
    print("\n")
    print("Orthogonal Matching Pursuit Accuracy: " + str(ompPred[0]) + " in " + \
        str(ompPred[2]) + " seconds")
    print("\n")

    # 2. Order results and find the best prediction in order to choose the best model
    results = [lasPred[0], mlbPred[0], kPred[0], ompPred[0]]
    greatest = results.index(max(results))

    # 3. based on the model selected, instatiate the model and then fit the data
    if greatest == 0:
        # Use Lasso regression if alpha does not equal zero. Otherwise use normal
        # linear regression because the smoothing out term in Lasso just goes to
        # to zero
        if lasPred[1] == 0:
            model = LinearRegression()
            model.fit(X_all, y)
        else:
            model = Lasso(alpha=lasPred[1])
            model.fit(X_all, y)
    elif greatest == 1:
        # Use mulinomial Naive Bayes
        model = MultinomialNB(alpha=mlbPred[1])
        model.fit(X_all, y)
    else:
        # Use KNearestNeighbor
        model = KNeighborsClassifier(n_neighbors=kPred[1])
        model.fit(X_all, y)

    return model

# PARAMETERS
#   path - <type 'str'> - This is the path of the file with the data that you want
# to prepare. Remember that it is important that it ends in .csv not some other
# file extension
# PURPOSE
#   The purpose of this function is to take in comma seperated data and then
# transform the text data and then split it up into testing and training data
# RETURNS
#   X_train_dtm - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that
# contains 3/4 of all the X's which will be used to train the models
#   y_train - <class 'pandas.core.series.Series'> - 3/4 of all the y's which will
# be used to train the models
#   X_test_dtm - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that
# contains 1/4 of all the X's which will be used to test the models
#   y_test - <class 'pandas.core.series.Series'> - 1/4 of all the y's wich will
# be used to test the models
#   X_all - <class 'scipy.sparse.csr.csr_matrix'> - Transformed text that will
# be used to fit the best optimized model in later functions
#   y - <class 'pandas.core.series.Series'> - All of the y's which will be used
# to fitthe best optimized model in later functions
#   vect - <class 'sklearn.feature_extraction.text.CountVectorizer'> - Vectorizer
# that is fitted with the data that we provided which will be used to transform all
# other text that we want to predict. One thing to remember here is that text that
# is not included in the original data set will not have any infulence on predictions
# with new data. For example, let's say that the word 'bad' never appears in our
# testing data but
def prepareTabulatedCategoricalData(path):
    # 1. Read in data
    data = pd.read_csv(path, names=['Category', 'Text'])

    # 2. Define X and y
    X = data.Text
    y = data.Category

    # 3. Split X and y into training and testing sets and define X_all
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # 4. Instatiate CountVectorizer to transform phrases into frequency sparse
    # matrices
    vect = CountVectorizer()

    # 5. Fit model with training data and then transform that data based on the
    # fit. Then, transform testing data based on the fit
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    X_all = vect.transform(X)

    return [X_train_dtm, y_train, X_test_dtm, y_test, X_all, y, vect, X]

# PARAMETERS
#   text - <type 'str'> - The text that you want to transform
#   vect - <class 'sklearn.feature_extraction.text.CountVectorizer'> - is a
# fitted vectorizer object that will be used to transform the text in to a count
# of particular words that were included when it was fit
#   model - - <class 'sklearn.Model'> - Where model is equal to the type of model
# that was passed. This could be anything from a KNearestNeighbor to Lasso Regression
# it just depends on the type of model that is passed to this function
# PURPOSE
#   The purpose of this function is to take a str, transform it using a fitted
# CountVectorizer which means that all the words that were in the set of data that
# was used to fit the CountVectorizer will be counted for this string. Then, with
# the transformed text, the function will try and predict the category and will
# return that category
# RETURNS
#   prediction - <type 'str'> - This is the category that the model precits based
# upon the text that was passed. This will actually be a number but it will return
# the string verson of the number in order for easy processing.
def predictText(text, vect, model):
    # 10. Define the question you want to predict and transform it
    phraseToTransform = vect.transform(text)

    prediction = str(model.predict(phraseToTransform)[0])

    return prediction

# PARAMETERS
#   path - <type 'str'> - The path of the file that you want to write to
#   categoryText - <type 'list'> - A list that contains the text and the associated
# category
#   X_all - <class 'pandas.core.series.Series'> - This is a pandas series that contains
# all of the text. This is important because it is used to make sure that the new
# text is not already written in the file. If it is already written in the file
# this function will ignore it and not write anything. The purpose of this is to
# prevent overfitting when you re-train the model.
# PURPOSE
#   The purpose of this function is to write the category and text to a specified
# file. While there is not a ton of checks (the function can easily be broken) it
# will not allow you to write text that is already written to the file. As mentioned
# above, this prevents overfitting.
def writeToFile(path, categoryText, X_all, alreadyWritten):
    # Check if the text already exists in the file you're trying to write to
    alreadyRecorded = False
    for x in X_all:
        if x == str(categoryText[1]):
            alreadyRecorded = True

    with open(path,"a") as csvfile:
        spamwriter = csv.writer(csvfile)
        if alreadyWritten == False:
            spamwriter.writerow([])
        if alreadyRecorded == False:
            spamwriter.writerow(categoryText)
        #csvfile.flush()
    csvfile.close

# PURPOSE
#   The purpose of this function is to prompt the user for the path of the data
# they would like to model. This function is breakable as well but it will check
# the path and make sure it is a valid path.
# RETURNS
#   path - <type 'str'> - This is the path of the file with the data you want to
# model
def promptUserForData():
    print("Please type the path of the file that contains the data you would like \
    to model")

    correctFilePath = False
    while correctFilePath == False:
        path = raw_input()
        print(path)
        try:
            with open(path, 'a') as test:
                spamwriter = csv.writer(test)
                spamwriter.writerow([])
            test.close
            break
        except OSError:
            print("Sorry! That was not a proper path. Could you try again?")

    print(type(path))
    return path

# PURPOSE
#   The purpose of the function is to prompt the user for the text they want to
# predict.
# RETURNS
#   text - <type 'list'> - This is just the text that was entered in a list. There
# is only one entry in the list but the reason it is a list is because predictText
# takes a list not a string. That is because CountVectorizer takes a list not a string
def promptUserForText():
    print("Please type the text you would like to predit")
    text = raw_input()

    # Put into list for predictText function
    text = [text]

    return text

# PURPOSE
#   The purpose of the function is to prompt the user to enter whether or not the
# file has already been written to. This is a weird write to csv file thing. If
# you use excel on a Mac to create a list of text and categories, the write csv
# function will not write to the next line. It will append to the last line first
# and then it will add a new line the next time you write to it. Kinda weird but
# I'm too lazy to investigate it further.
# RETURNS
#   alreadyWritten - <type 'bool'> - This is a boolean that the is entered by the
# user on whether or not the file that is being written to has ever been written
# to by this program before
def promptUserIfFileWasAlreadyWritten():
    alreadyWritten = False
    print("Is this the FIRST time you have written to this file? If you've \
        never written to this file then type or say 'yes'. If you have written \
        written to this file already then type of say 'no'")
    input = raw_input()
    if input == "no" or input == "No":
        alreadyWritten = True

    return alreadyWritten

# PARAMETERS
#   prediction - <type 'str'> - This is the category prediction. It is actually
# a number but I converted it to a string just so it would be easy to print
# out.
#   phrase - <type 'list'> - This is a list that contains the string that was
# predicted. That's all it contains. Eventually you could scale this so that if
# the list contains more than just one entry, you could go through each one and
# write to the file or prompt the user for the correct category
#   path - <type 'str'> - The path of the file that you want to write to
#   X - <class 'pandas.core.series.Series'> - A pandas data series that contains
# all of the text that is already in the file. The purpose of this is to use it
# to check if the text that you just predicted already exists. If it does, you
# would not want to write the same text again. This all happens in the writeToFile
# function not this function but this function calls the writeToFile function
# PURPOSE
#   Ask the user if the prediction is correct. If yes, then add the
# prediction to the csv file. If no, then prompt the user to enter the
# correct prediction into the terminal and then write that to the csv file
def promptUserForCorrectionOfPrediction(prediction, phrase, path, X):
    print("Is this correct? " + prediction)
    input_line = raw_input()
    if input_line == "yes":
        yes_text = [prediction, phrase[0]]

        # Write to file
        writeToFile(path, yes_text, X, True)

    elif input_line == "no":
        # Prompt user to enter correct category so that it can develop better
        # model
        print("What is the correct category? " + str(y.unique()))
        input_category = raw_input()
        input_cateogry = int(input_category)

        # Create entry based on the category entered and the phrase used
        no_text = [str(input_category), phrase[0]]

        # Write to file
        writeToFile(path, no_text, X, True)

    else:
        # used only for testing purposes
        print("other")


def main():
    # Prepare the data
    path = promptUserForData()
    data = prepareTabulatedCategoricalData(path)

    # Find the best model
    model = findBestOptimizedModel(data[0], data[1], data[2], data[3], data[4], data[5])

    # Takes text, CountVectorizer object and a model and predicts category
    phrase = promptUserForText()
    prediction = predictText(phrase, data[6], model)

    # Terminal Style
    print("\n")
    print("\n")
    print("\n")
    print("\n")

    # Test to see if csv file has already been written to. If it has not, you
    # need to write an empty line to the file (e.g. spamwriter.writerow([]))
    # so that it does not write on the last line of data in the excel sheet
    alreadyWritten = promptUserIfFileWasAlreadyWritten()

    promptUserForCorrectionOfPrediction(prediction,phrase,path,data[7])


if __name__ == "__main__":
    main()

# Confusion Matrix -> array([[true positive, false positive], [false negative, true negative]])
# Make sure the optimizing models use all the data, not just the training data
# Does it make sense to find the optimal parameter with all of the data
# line 408 where it fits and transforms again. that's wrong lol
