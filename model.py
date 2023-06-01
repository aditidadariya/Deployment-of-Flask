#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:02:18 2023

@author: aditidadariya
"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
#from pandas import read_csv
from matplotlib import pyplot
import seaborn as sns
# Import matplotlib.pyplot to draw and save plots
import matplotlib.pyplot as plt
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
#import config.py
from config import *
#import functionlibrary.py
from functionlibrary import *
import time


######################### STEP 2: READING THE DATA ############################


# Calling ReadDataset function from functionlibrary.py file. 
# It reads the dataset file [1] and store it in a dataframe along with its attributes
# location_of_file is defined in the config.py file
dfcredit = ReadDataset(location_of_file)
print(dfcredit.head(5))
# Print the number of rows and columns present in dataset
print("The dataset has Rows {} and Columns {} ".format(dfcredit.shape[0], dfcredit.shape[1]))

# Give a new line to clearly format the output
Newline()

# Output:
 #   A1     A2     A3 A4 A5 A6 A7    A8 A9 A10  A11 A12 A13    A14  A15 A16
 # 0  b  30.83  0.000  u  g  w  v  1.25  t   t    1   f   g  00202    0   +
 # 1  a  58.67  4.460  u  g  q  h  3.04  t   t    6   f   g  00043  560   +
 # 2  a  24.50  0.500  u  g  q  h  1.50  t   f    0   f   g  00280  824   +
 # 3  b  27.83  1.540  u  g  w  v  3.75  t   t    5   t   g  00100    3   +
 # 4  b  20.17  5.625  u  g  w  v  1.71  t   f    0   f   s  00120    0   +
 # The dataset has Rows 690 and Columns 16 

##################### STEP 3: PRE-PROCESSING THE DATA #########################

#======================== 3.1. Understand the data ============================

# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print(dfcredit.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# only on columns A3, A8, A11, A15, which specifies that other columns are having 
# some unnecessary data. 
# To get rid of missing values, Step 3.2 is performed

# Give a new line to clearly format the output
Newline()

#               A3          A8        A11            A15
#count  690.000000  690.000000  690.00000     690.000000
#mean     4.758725    2.223406    2.40000    1017.385507
#std      4.978163    3.346513    4.86294    5210.102598
#min      0.000000    0.000000    0.00000       0.000000
#25%      1.000000    0.165000    0.00000       0.000000
#50%      2.750000    1.000000    0.00000       5.000000
#75%      7.207500    2.625000    3.00000     395.500000
#max     28.000000   28.500000   67.00000  100000.000000

#======================= 3.2. Find the missing values =========================

# Printing the missing values (NaN)
print(dfcredit.isnull().sum())

# Give a new line to clearly format the output
Newline()

# Output:
#A1     0
#A2     0
#A3     0
#A4     0
#A5     0
#A6     0
#A7     0
#A8     0
#A9     0
#A10    0
#A11    0
#A12    0
#A13    0
#A14    0
#A15    0
#A16    0
#dtype: int64   

# The output shows that there are no missing values (NaN) avaliable in the data

# To get a clear view of this kind of data, dataset is verified manually and found that 
# there are "?" special character specified in many columns
# Therefore Step 3.3 will find the rows having "?" and drop the respective row from the dataset

#================== 3.3. Find and drop unnecessary values =====================

# Finding the special character "?" in the dataset and droping the rows, to get a clean dataset [4]

# Calling RemoveSpecialChar function from functionlibrary.py file to remove all the special characters from dataset
dfcredit = RemoveSpecialChar(special_char,dfcredit)

# Summarizing the dataset by printing the number of rows and columns present in dataset
print("The updated dataset has Rows {} and Columns {} ".format(dfcredit.shape[0], dfcredit.shape[1]))

# Give a new line to clearly format the output
Newline()

# Summarizing the data statisticaly again by getting the mean, std and other values for all the columns
print(dfcredit.describe())

# The output again shows that mean, std etc is done on attributes A3, A8, A11 and A15 only, instead of all the attributes

# Give a new line to clearly format the output
Newline()

# Verify the data types of all attributes in the dataset, to verify the difference between the datatypes
print(dfcredit.info())

# The information shows that the datatype of attributes are of int, float and object types
# The data needs to be converted to have consistency among all the attributes, 
# because the dataset does not have a clear instruction of what all the attributes signifies to.
# Therefore, it becomes necessary to have the data of same date types

# Give a new line to clearly format the output
Newline()

#======= 3.4. Analysing and making consistent data type for all attributes ======

# Convert the column values from string/object into integer using labelencoder [5]

# In the dataset, there are mix the data types, such as values as string, float and integer
# To have same datatype, created Encoder function to convert the strings into integer values
     
# Calling the Encoder function passing the dataframe as parameter that creats a new dataframe to store the new values
dfcredit = Encoder(dfcredit)

# Save the new dataframe into a csv file
df =  pd.DataFrame(dfcredit)
df.to_csv('cleancreditdata.csv')

# Give a new line to clearly format the output
Newline()

# Summarizing the data statisticaly again by getting the mean, std and other values on all the columns
print(dfcredit.describe())
print(dfcredit.info())
# Now the count, mean, std, min, max etc are shown for all attributes, this means the data is clean now

# Give a new line to clearly format the output
Newline()

#======================== 3.5. Class distribution =============================

# As mentioned in the dataset details, "A16" attribute is the class attribute [1]

# Finding the number of rows that belong to each class
print(dfcredit.groupby('A16').size())

# Output is showing that the class "0" is having 296 rows and class "1" is having 357 rows
# By this we understand that there are only 2 classes 0 and 1 available in the data
# which means this is a Binary Classrification type.
# This also show the 0 is assigned to denied requests for applications asking credit card approval
# and 1 is assigned for approved requests for applications asking credit card approval
# This also states that the data is imbalanced. 
# To understand the dataset more, Data Visualization by ploting the pair plot and heat graph is done in next step

# Give a new line to clearly format the output
Newline()

####################### Step 4: DATA VISUALIZATION ############################

#=========================== 4.1. Plot Pairplot ===============================

# To visualize the data, pair plot has been demonstrated below [6]

# Clear the plot dimensions
plt.clf()

# Plot the multivariate plot using pairplot
#sns.set(rc={'figure.figsize':(10,10)})
sns.set(style="ticks", color_codes=True)
grh = sns.pairplot(dfcredit,diag_kind="kde")
plt.show()

# Give a new line to clearly format the output
Newline()

# Wait for the graph to be displayed
time.sleep(30)

# The plots suggests that there is a high correlation and a predictable relationship 
# between the diagonal grouping of some pairs of attributes
# As stated earlier that the data is imbalanced, the graph is not clearly linear too.

#=========================== 4.2. Plot Heat map =============================

# To visualize the data again, heat map is demonstrated below [7]

# Clear the plot dimensions
plt.clf()

# Plot Heat graph to visualise the correlation
plt.figure(figsize=(20,10))
matrix1=dfcredit.corr()
plt.title('Credit Approval Correlation', size = 15)
sns.heatmap(matrix1, vmax=10.8, square=True, cmap='YlGnBu', annot=True)

# Save the pairplot in png file
#plt.savefig("heatplot.png")

# Give a new line to clearly format the output
Newline()

# Wait for the graph to be displayed
time.sleep(30)

# As observed earlier in the pairplot, the data has high correlation and a predictable relationship 
# between the diagonal grouping of some pairs of attributes. 
# The same has been observed in Hear map as well.
# To fix the imbalance problem, I have used the balancing technique RandomUnderSampler and SMOTE
# in the further steps after checking the performance of Decision Tree and Linear Discriminent Analysis models
# using the basic HoldOut Validation (i.e. train and test split)

######################## STEP 5: FEATURE SELECTION ############################

# To implement any model on the data, we need to do feature selection first
# and therefore the dataset is devided into 2 parts.
# Firstly features (all the attributes except the target attribute) and secondly target (class attribute).
# The dfcredit dataframe has been split into dfcredittarget having only column A16, as this is the class attribute,
# and all the other attributes are taken into dfcreditdata dataframe as features

dfcreditdata = dfcredit.drop("A16", axis=1) # Features
dfcredittarget = dfcredit.iloc[:,-1]    # Target

# None of the feature selection methods are used here because the dataset is having very less number of attributes.
# and therefore, the dimentionality reduction has not been performed here.
# Also, the attirbute names have been changed to meaningless by the author [1], hence all the attributes are considered as features

# $$$$$$$$$$$$$$$$$$$$$$ BUILDING DECISION TREE CLASSIFIER AND LINEAR DISCRIMINENET ANALYSIS MODELS $$$$$$$$$$$$$$$$$$$$$$$$$$$

############## STEP 6: BUILD BASE MODELS WITH HOLDOUT VALIDATION ###################

# Base Model is built below using the HoldOut Validation (train test split) to evaluate the accuracy

# ================ 6.1. Split the data to train and test set ==================

# Spot Check Algorithms with Decision Tree [8] [9] [10] and Linear Discriminant algorithms [11] [12]

# Data is splited into training and test set using the feature and target dataframes
# to understand the model performance with 70% training set and 30% test set
X_train, X_test, Y_train, Y_test = train_test_split(dfcreditdata, dfcredittarget, test_size=0.3, random_state=None)

# ========================== 6.2. Define models ===============================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier [8] [9] [10]
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis [11] [12]
models.append(('LDA', LinearDiscriminantAnalysis()))

# ===================== 6.3. Build Model and Evaluate it  =====================

# Calling BasicModel function
BasicModel(models,X_train,Y_train,X_test,Y_test)

# Create a dataframe to store accuracy
dfbasicscore = pd.DataFrame(basic_score)    
print(dfbasicscore.head())

# Give a new line to clearly format the output
Newline()

# Output:
    # On DTCLS Accuracy is: 81.632653 
    # On LDA Accuracy is: 85.204082 
# The output shows that the accuracy for LDA model is better than Decision Tree Classifier.
# To improve the performance of each model, the evaluation is done with respect to different random_state in the next step


################## STEP 7: BUILD MODELS WITH RANDOM STATE #####################

# Model is build with random state to evelaute the accuracy

# ======================= 7.1. Define models ==================================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))


# ===================== 7.2. Build Model and Evaluate it  =====================

# Calling BuildModelRS to evaluate the models with random states and return the accuracy
BuildModelRS(models,rand_state,dfcreditdata,dfcredittarget)

# Create a dataframe to store accuracy
dfrsscore = pd.DataFrame(score)    
print(dfrsscore.head(8))

# Give a new line to clearly format the output
Newline()

# The output shows that the accuracy changes in each randon state. Therefore, data has been balanced in further steps.

######################## STEP 8: BALANCING THE DATA ###########################

# Installed imbalance-learn library using "conda install -c conda-forge imbalanced-learn" [15]

# To balance the data, RandomUnderSampler and SMOTE is utilized to undersample and oversample the data along with pipeline [16]

# Define oversmaple with SMOTE function
oversample = SMOTE()
# Define undersample with RandomUnderSampler function
undersample = RandomUnderSampler()
# Define Steps for oversample and undersample
steps = [('o', oversample), ('u', undersample)]
# Define the pipeline with the steps
pipeline = Pipeline(steps = steps)
# Fit the features and target using pipeline and resample them to get X and Y
X, Y = pipeline.fit_resample(dfcreditdata, dfcredittarget)
# Print the shape of X and Y
#print("The features dataset has Rows {} and Columns {} ".format(X.shape[0], X.shape[1]))
#print("The target dataset has Rows {} and Columns {} ".format(Y.shape[0],0))

# Give a new line to clearly format the output
Newline()

# By balancing the data using oversample and undersample, X and Y now have adequate amount of data avaialble
# Reducing the dimensionality is not needed because the dataset is small with only 15 features


######## STEP 9: OPTIMIZE MODEL ON BALANCED DATA USING CROSS VALIDATION #######

# StratifiedKFold Cross Validation is utilized to improve the performance as it splits the data approximately in the same percentage [17] [18]
# StratifiedKFold Cross Validation is used because there are 2 classes, and the split of train and test set could be properly done

# ============================ 9.1. Define models =============================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))

# ===================== 9.2. Build Model and Evaluate it  =====================

# Calling BuildModelBalCV function to get the accuracy, cross validation results and names of models used
BuildModelBalCV(models,X,Y)
# Create a dataframe to store accuracy
dfscore = pd.DataFrame(score)    
print(dfscore.head())

# Give a new line to clearly format the output
Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Give a new line to clearly format the output
Newline()

# Outout:
    # On DTCLS: Mean is 82.355243 and STD is 0.033680
    # On LDA: Mean is 87.255477 and STD is 0.040058

# LDA is having the highest Mean value of 87.25%.
# Decision Tree has also performed good with Mean as 82.35%
# Looking at the boxplot and the Mean values, it is clear that LDA has performed better even after balancing the data 
# along with StratifiedKFold Cross Validation by giving Mean as 87.25%


################## STEP 10: TUNE DECISION TREE BY GRIDSEARCHCV ##################

# Tuning the Decision Tree Classifier with GridSearchCV to find the best hyperparameter [19] [20]

# Define decision tree classifier model
dtcmodel = DecisionTreeClassifier()
# Define StratifiedKFold
skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
# Define parameters
param_dict = dict()
param_dict = {"criterion": ["gini", "entropy"], "max_depth": [7,5,3]}
# Build GridSearchCV to get the accuracy
search = GridSearchCV(dtcmodel, param_dict, scoring='accuracy', cv=skfold, n_jobs=-1)
# Fit the GridSearchCV
results = search.fit(X, Y)
# Summarize 
dt_accuracy = results.best_score_
dt_para = results.best_params_
print('Decision Tree Mean Accuracy: %f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Give a new line to clearly format the output
Newline()

# Output:
    # Decision Tree Mean Accuracy: 0.865630
    # Config: {'criterion': 'entropy', 'max_depth': 3}
# The best parameter for Decision Tree Classifier are 'criterion' as 'entropy', 'max_depth' as 3, 
# using which the performance of decision tree has been increased slightly with 86.56%

###################### STEP 11: TUNE LDA BY GRIDSEARCHCV ######################

# Tuning the LDA with GridSearchCV to find the best hyperparameter [19]

# Define LDA model
ldamodel = LinearDiscriminantAnalysis()
# Define StratifiedKFold
skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
# Define parameters
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# Build GridSearchCV to get the accuracy
search = GridSearchCV(ldamodel, grid, scoring='accuracy', cv=skfold, n_jobs=-1)
# Fit the GridSearchCV
results = search.fit(X, Y)
# Summarize
lda_accuracy = results.best_score_
lda_para = results.best_params_
print('LDA Mean Accuracy: %f' % results.best_score_)
print('Config: %s' % results.best_params_)
#print(lda_accuracy)
#print(lda_para['solver'])

# Give a new line to clearly format the output
Newline()

# Output:
    # LDA Mean Accuracy: 0.871088
    # Config: {'solver': 'svd'}

# The best parameter LDA are 'solver' as 'svd' 
# using which the performance of LDA model has been increased slightly to 87.10%


######################### STEP 12: BUILD TUNED MODEL ##########################

# Building the model finally on balanced data with best hyper-parameters along with StratifiedKFold cross validation [21]

# ========================== 12.1. Define models ==============================

# Clear lists
ClearLists()

# Define models with the best hyperparameters found earlier
models.append(('DTCLS', DecisionTreeClassifier(criterion=dt_para['criterion'], max_depth=dt_para['max_depth'])))
models.append(('LDA', LinearDiscriminantAnalysis(solver=lda_para['solver'])))

# ==================== 12.2. Build Model and Evaluate it  =====================

# Calling BuildFinalModel to get the accuracy after tuning
BuildFinalModel(models,X,Y)
# Create a dataframe to store accuracy
dffinalscore = pd.DataFrame(score) 
print(dffinalscore.head())

max_mean = dffinalscore['Mean'].max()
max_mean_indx = dffinalscore['Mean'].idxmax()
model_name = dffinalscore.loc[max_mean_indx, 'Model Name']

# Give a new line to clearly format the output
Newline()

print('%s has the max accuracy as %f' % (model_name, max_mean))

# Give a new line to clearly format the output
Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(final_results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Give a new line to clearly format the output
Newline()

# Output:
    # On DTCLS: Mean is 84.178404 and STD is 0.035138
    # On LDA: Mean is 86.977700 and STD is 0.029924
    
# Looking at the Mean and Boxplot, it is clear that LDA has performed better than 
# Decision Tree after tuning the model with hyperparameters using StratifiedKFold cross validation. 
# However, there is not much difference in both the accuracies.

######################### STEP 12: BUILD BEST PERFORMED MODEL ##########################


# Clear lists
ClearLists()


# Define models with the best hyperparameters found earlier
if model_name == "DTCLS":
    models.append(('DTCLS', DecisionTreeClassifier(criterion=dt_para['criterion'], max_depth=dt_para['max_depth'])))
elif model_name == 'LDA':
    models.append(('LDA', LinearDiscriminantAnalysis(solver=lda_para['solver'])))

# Calling BuildFinalModel to get the prediction after tuning
model, score, final_results = BuildFinalModel(models,X,Y)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     MODEL.PKL FILE     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Dump the model.pkl file
import pickle
pickle.dump(model, open('model.pkl', 'wb'))


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ REFERNCES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# [1] https://archive.ics.uci.edu/ml/datasets/Credit+Approval
# [4] https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
# [5] https://www.projectpro.io/recipes/convert-string-categorical-variables-into-numerical-variables-using-label-encoder
# [6] https://pythonbasics.org/seaborn-pairplot/
# [7] https://seaborn.pydata.org/generated/seaborn.heatmap.html
# [8] https://scikit-learn.org/stable/modules/tree.html
# [9] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# [10] https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# [11] https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
# [12] https://www.statology.org/linear-discriminant-analysis-in-python/
# [13] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# [14] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# [15] https://anaconda.org/conda-forge/imbalanced-learn
# [16] https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# [17] https://www.kaggle.com/vitalflux/k-fold-cross-validation-example/
# [18] https://scikit-learn.org/stable/modules/cross_validation.html
# [19] https://machinelearninghd.com/gridsearchcv-classification-hyper-parameter-tuning/
# [20] https://ai.plainenglish.io/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda
# [21] https://www.kaggle.com/vitalflux/k-fold-cross-validation-example/

