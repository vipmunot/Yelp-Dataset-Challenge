# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:09:04 2016

@author: Vipul Munot
Link: https://github.com/karenxiao/pandas-express
"""
import json
import pandas as pd
def load_data(filepath):
    data = []
    with open(filepath) as file:
        for line in file:
            data.append(json.loads(line.rstrip()))
    return data


def main():
    business = pd.DataFrame.from_dict(load_data("E:/IUB/Search/Project/business.json"))
    # Create dummy variables for categories
    categories_df = business['categories'].str.join(sep=',').str.get_dummies(sep=',')
    # Save the list of categories for future use
    categories = categories_df.columns.values
    # Merge it with our original dataframe
    business = pd.merge(business, categories_df, left_index = True, right_index = True)
    #Instead of dropping the categories column, we're going to keep it around, but reformat it as a tuple
    business['categories'] = business['categories'].apply(lambda x: tuple(x))
    print(business['categories'])
#    print(business[business['Chinese'] == 1].head())
#    print(business['Chinese'].sum())
#    review = pd.DataFrame.from_dict(load_data("E:/IUB/Search/Project/review.json"))

if __name__ == '__main__':
	main()