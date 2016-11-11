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
	business = pd.DataFrame.from_dict(load_data("E:/IUB/Search/Project/business.json"))
	review = pd.DataFrame.from_dict(load_data("E:/IUB/Search/Project/review.json"))
	tip = pd.DataFrame.from_dict(load_data("E:/IUB/Search/Project/tip.json"))
	business = business[['business_id','categories']]
	review = review[['business_id','text']]
	tip= tip[['business_id','text']]
	reviewData = pd.merge(review, business, on='business_id')
	reviewData.columns = ['business_id','review','categories']
	tipData = pd.merge(tip, business, on='business_id')
	tipData.columns = ['business_id','tip','categories']
	print("Tip Data:\t",tip.shape)
	print("Review Data:\t",review.shape)
	print("Business Data:\t",business.shape)
	print("Final Review Data:\t",reviewData.shape)
	print("Final Tip Data:\t",tipData.shape)
	del review
	del business
	del tip

	
def main():
    preprocessing()
    

if __name__ == '__main__':
	main()
