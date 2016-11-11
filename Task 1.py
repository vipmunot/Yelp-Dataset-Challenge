# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:09:04 2016

@author: Vipul Munot
"""
import re
import warnings
import pandas as pd
from bs4 import BeautifulSoup
import mongo
warnings.filterwarnings("ignore")

def convert_words( raw_review ):
    review_text = BeautifulSoup(raw_review,'lxml').get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    return( " ".join( words )) 

''' 
Filtering Categories
''' 
business =  pd.DataFrame(list(mongo.mongo_business))
business = business[['business_id','categories']]
''' 
Filtering Tips
''' 
tipData =  pd.DataFrame(list(mongo.mongo_tip))
tipData = tipData[['business_id','text']]
tipData.columns = ['business_id','tip']
tipData.loc[:,'tip'] = tipData['tip'].map(convert_words)

''' 
Merging Categories and Tips
''' 
tip = pd.merge(tipData, business, on='business_id')

''' 
Filtering Categories
''' 
reviewData =  pd.DataFrame(list(mongo.mongo_review))
reviewData = reviewData[['business_id','text']]
reviewData.columns = ['business_id','review']
reviewData.loc[:,'tip'] = reviewData['review'].map(convert_words)
''' 
Merging Categories and Reviews
''' 
review = pd.merge(reviewData, business, on='business_id')    