#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 00:06:10 2023

@author: aditidadariya
"""

############################# VARIABLE DECLARATION ############################

import os

# location_of_file variable is defined to store the dataset file location
absolute_path = os.path.dirname(__file__)
relative_path = "Dataset/crx.csv"
location_of_file = os.path.join(absolute_path, relative_path)

# special_char variable is defined to store any special characters
special_char = "?"

# Define models list to store the model object
models = []

# Define names list to store the name of models
names = []

# Define results list to store the accuracy score
results = []

# Define basicscore list to store the name and accuracy score
basic_score = []

# Define basicscore list to store the name and accuracy score
score = []

# Define finalresultslist to store cross validation score
final_results = []

# Defined a randstate variable to store the input for random_state in train_test_split function later
rand_state = [1,3,5,7]

