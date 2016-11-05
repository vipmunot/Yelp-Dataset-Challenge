# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:09:04 2016

@author: Vipul Munot
"""
from pymongo import MongoClient
import pandas as pd
client =  MongoClient()
db = client.Yelp
collection = db.review
print ("Total Rows: ",collection.count())
for row in collection.find():
    data = pd.DataFrame(row)
print(data['stars'].value_counts())
