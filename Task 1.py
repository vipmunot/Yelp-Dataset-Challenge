# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:23:07 2016

@author: Vipul Munot
"""

import pandas as pd


from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

review = pd.read_pickle('final.pkl')
'''
Splitting the data
'''
X_train, X_test, y_train, y_test = train_test_split(review["review"], review["categories"], test_size=0.20, random_state=4212)
'''
Applying TF-IDF to the data
'''
vectorizer = TfidfVectorizer() 
tfidfXtrain = vectorizer.fit_transform(X_train)
tfidfXtest = vectorizer.transform(X_test)
'''
Random Forest
'''
forest = RandomForestClassifier(max_features=500,n_estimators=100, n_jobs=4)
forest = forest.fit( tfidfXtrain, y_train )
result = forest.predict(tfidfXtest)
ooutput = pd.DataFrame( data={"predicted":result,"actual":y_test,'review':X_test} )
string = output.iloc[0]['review']
print(output.head())
print ("accuracy_score: ", accuracy_score(y_test.values,result))
for index, row in output.iterrows():
    if row['review'] in string:
        print(row['predicted'])
for index, row in review.iterrows():
    if row['review'] in string:
        print(row['categories'])