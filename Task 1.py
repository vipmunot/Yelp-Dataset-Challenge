# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:23:07 2016

@author: Vipul Munot
"""

import pandas as pd


from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

review = pd.read_pickle('review.pkl')

X_train, X_test, y_train, y_test = train_test_split(review["review"], review["categories"], test_size=0.20, random_state=4212)
vectorizer = TfidfVectorizer()
f_train = vectorizer.fit_transform(X_train)
