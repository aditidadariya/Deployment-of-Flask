#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:19:43 2023

@author: aditidadariya
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction == [0]:
        output = "Not Approved"
    elif prediction == [1]:
        output = "Approved"
    return render_template("index.html", prediction_text = "The predicted approval is {}".format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    if prediction == [0]:
        output = "Not Approved"
    elif prediction == [1]:
        output = "Approved"
    return jsonify(output)
    
if __name__ == "__main__":
    app.run(debug=True)
    
