import sys

import TopicModeller as tm
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
import unicodedata
import codecs


class BusinessReview:
    def __init__(self):
        self.business_id = ""
        self.review_list = []
        self.tip_list = []
        self.positive = []
        self.negative = []

def main():
    fname_p = 'Task 2/review/positive.txt'
    positive_reviews = codecs.open(fname_p, 'r', encoding='utf-8', errors='ignore').readlines()
    tm.build_topic_model(positive_reviews, 'positive')

'''
    fname_n = 'Task 2/review/negative.txt'
    negative_reviews = codecs.open(fname_n, 'r', encoding='utf-8', errors='ignore').readlines()
    tm.build_topic_model(negative_reviews, 'negative')
'''
if __name__ == '__main__':
    main()
