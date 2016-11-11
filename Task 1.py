# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:09:04 2016

@author: Vipul Munot
"""
import json
import pandas as pd
def load_data(filepath):
    data = []
    with open(filepath) as file:
        for line in file:
            data.append(json.loads(line.rstrip()))
    return data
def preprocessing():
    business = pd.DataFrame.from_dict(load_data("business.json"))
    review = pd.DataFrame.from_dict(load_data("review.json"))
    tip = pd.DataFrame.from_dict(load_data("tip.json"))
    business = business[['business_id','categories']]
    review = review[['business_id','text']]
    tip= tip[['business_id','text']]
    review_business = pd.merge(review, business, on='business_id')
    review_business.columns = ['business_id','review','categories']
    data =  pd.merge(tip, review_business, on='business_id')
    print(data.head(2))
    data.to_json('yelp_data.json')
def main():
    preprocessing()
    

if __name__ == '__main__':
	main()
