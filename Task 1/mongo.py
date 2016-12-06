# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:41:20 2016

@author: Vipul Munot
"""

from pymongo import MongoClient

client = MongoClient()


db = client['Yelp']
mongo_review = db['review'].find({})
mongo_tip = db['tip'].find({})
mongo_business = db['business'].find({})

