#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 00:44:49 2023

@author: aditidadariya
"""

import requests
url = 'http://localhost:5000/predict_api'

r = requests.post(url,json={"A1":1,
   "A2":153,
   "A3":0,
   "A4":1,
   "A5":0,
   "A6":12,
   "A7":7,
   "A8":1.25,
   "A9":1,
   "A10":1,
   "A11":1,
   "A12":0,
   "A13":0,
   "A14":68,
   "A15":0
})

print(r.json())
