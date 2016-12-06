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

def replicateClasses(review):
    cols = review.columns
    categories = review["categories"]    
    data = pd.DataFrame(columns=cols)   
    r_index = 0
    count = 0
    for i in range(len(categories)):
        for item in categories[i]:
            # print(item)
            data.loc[len(data)-1] = [review["business_id"][r_index], review[cols[1]][r_index], item]
        r_index += 1
        count += len(categories[i])
    print("Total Categories:\t",count)
    return (data)    
''' 
Filtering Business
''' 
business =  pd.DataFrame(list(mongo.mongo_business))
business = business[['business_id','categories']]
print("Business Filtered")
''' 
Filtering Tips
''' 
tipData =  pd.DataFrame(list(mongo.mongo_tip))
tipData = tipData[['business_id','text']]
tipData.columns = ['business_id','tip']
tipData.loc[:,'tip'] = tipData['tip'].map(convert_words)
print("Tips Filtered")
''' 
Merging Business and Tips
''' 
tip = pd.merge(tipData, business, on='business_id')
print("Tips Merged")
tip = replicateClasses(tip)
tip.to_pickle('tip.pkl')
print("Tips Pickle Created")
''' 
Filtering Categories
''' 
reviewData =  pd.DataFrame(list(mongo.mongo_review))
reviewData = reviewData[['business_id','text']]
reviewData.columns = ['business_id','review']
reviewData.loc[:,'review'] = reviewData['review'].map(convert_words)
print("Review Filtered")
''' 
Merging Categories and Business
''' 
review = pd.merge(reviewData, business, on='business_id')   
print("Review Merged")
review = review[:30000]
review = replicateClasses(review)
review.to_pickle('review.pkl')
print("Tips Pickle Created")