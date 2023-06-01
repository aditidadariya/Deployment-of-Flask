#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 00:07:18 2023

@author: aditidadariya
"""

######################## Importing necessary libraries ########################
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
# Import matplotlib.pyplot to draw and save plots
import matplotlib.pyplot as plt
# Import LabelEncoder to change the data types of attributes
from sklearn.preprocessing import LabelEncoder
# Import parameters.py file to get all the variables here
#import config.py
from config import *
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import cross_val_score function
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# Import StratifiedKFold function
from sklearn.model_selection import StratifiedKFold
# Import GridSearchCV function
from sklearn.model_selection import GridSearchCV
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
# Import confusion_matrix function
from sklearn.metrics import confusion_matrix
# Import classification_report function
from sklearn.metrics import classification_report
# Import ConfusionMatrixDisplay function to plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Import render from graphviz to convert dot file to png
from graphviz import render
# Import RandomUnderSampler, SMOTE, Pipeline to balance the dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


########################## Function declarations ##############################

# Newline gives a new line
def Newline():
    print("\r\n")
    
# ClearLists removes all values from lists
def ClearLists():
    models.clear()
    names.clear()
    results.clear()
    basic_score.clear()
    score.clear()
    final_results.clear()
    


# ReadDataset function reads the dataset csv file and store it in a dataframe
# location_of_file is variable thats stores the file location and is defined in config.py file
def ReadDataset(location_of_file):
    # Read the data file from specified location
    df = pd.read_csv(location_of_file,header=None)
    # Specifying the column names into the dataset using dataframe, as there are no columns already specified in the dataset
    df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
    return df


# for loop will iterate over all the columns in dataframe to find and drop the rows with special characters [4]
def RemoveSpecialChar(special_char,df):
    for eachcol in df:
        # Drop rows where "?" is displayed
        df.drop(df.loc[df[eachcol] == special_char].index, inplace=True)
    return df


# Encoder function transforms the character and object value in dataframe [5]
def Encoder(df): 
    # columnsToEncode is to get a list of columns having datatype as category and object
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    # le is an object of LabelEncoder with no parameter
    le = LabelEncoder()
    # for loop will iterate over the columns. 
    # it will first try to use LabelEncoder to fit_transform
    # and if there are any error than the except will be executed
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature) 
    return df


# Define BasicModel function to evaluates the models
# displays the confusion matrix and plot it [13] [14]
# displays the classification_report
def BasicModel(models,X_train,Y_train,X_test,Y_test):
    # Evaluate each model in turns
    for name, model in models:
        # Train the model
        modelfit = model.fit(X_train,Y_train)
        # Predict the response for test dataset
        Y_predict = modelfit.predict(X_test)
        # Store the accuracy in results
        results.append(metrics.accuracy_score(Y_test, Y_predict))
        # Store the model name in names
        names.append(name)
        # Print the prediction of test set
        print('On %s Accuracy is: %f ' % (name, metrics.accuracy_score(Y_test, Y_predict)*100))
        # Store the name and accuracy in basic_score list
        basic_score.append({"Model Name": name, "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
        # Print Confusion Matrix and Classification Report
        print(confusion_matrix(Y_test, Y_predict))
        print(classification_report(Y_test, Y_predict))
        # Plot Confusion Matrix [13] [14]
        cm = confusion_matrix(Y_test, Y_predict, labels=modelfit.classes_)
        cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelfit.classes_)
        cmdisp.plot()
        plt.show()
    return basic_score

# Define BuildModelRS function to evaluate the models based on the random states [8-12]
# rand_state variable has been defined in the config.py file with values 1,3,5,7
# stores the Model Name, Random State and Accuracy of each model in score list
def BuildModelRS(models,rand_state,dfcreditdata,dfcredittarget):
    # Evaluate each model in turn
    for name, model in models:
        # for loop will train and predict the decision tree model on different random states
        for n in rand_state:
            # The training set and test set has been splited using the feature and target dataframes with different random_state
            X_train, X_test, Y_train, Y_test = train_test_split(dfcreditdata, dfcredittarget, test_size=0.3, random_state=n)
            # Train Decision Tree Classifer
            modelfit = model.fit(X_train,Y_train)
            # Predict the response for test dataset
            Y_predict = modelfit.predict(X_test)
            # Store the accuracy in results
            results.append(metrics.accuracy_score(Y_test, Y_predict))
            # Store the model name in names
            names.append(name)
            # Store the Model Name, Random State and Accuracy into score list
            score.append({"Model Name": name, "Random State": int(n), "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
    return score

# Define function BuildModelBalCV to evaluate models on the balanced data and utilize cross validation
def BuildModelBalCV(models,X,Y):
    # Evaluate each model in turn  
    for name, model in models:
        # Define StratifiedKFold [17] [18]
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        model = model.fit(X,Y)
        # Store the accuracy in results
        results.append(cv_results)
        # Store the model name in names
        names.append(name)
        # Print the results
        print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean()*100, cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Mean": cv_results.mean()*100, "STD": cv_results.std()})
    return score
    return results
    return names

# Define BuildFinalModel function to evaluate models with their hyper-parameters
def BuildFinalModel(models,X,Y):
    # Evaluate each model in turn 
    for name, model in models:
        # Define StratifiedKFold
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation [21]
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        model = model.fit(X.values,Y)
        # Perform cross-validation and obtain predictions
        #predictions = cross_val_predict(model, X, Y, cv=5)        #predictions = model.fit(X, Y).predict(X)
        # Store the cross validationscore into results
        final_results.append(cv_results)
        # Store model name into names
        names.append(name)
        # Print the results
        #print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean(), cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Mean": cv_results.mean(), "STD": cv_results.std()})
        #print('predictions are ')
        #print(predictions)
    return model, score, final_results 
    #return model, score, final_results, predictions 


