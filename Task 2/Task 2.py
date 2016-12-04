import sys

import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords

class BusinessReview:
    def __init__(self):
        self.business_id = ""
        self.review_list = []
        self.tip_list = []
        self.positive = []
        self.negative = []


def analyze_reviews(review_list):
    count = 0
    for one_review in review_list:
    	# if "sentiment" not in one_review:
    	if True:
            r  = TextBlob(one_review).sentiment.polarity
            print('review = ', one_review)
            print( 'sentiment = ',r )
    	    #sentimentDict[one_review] = one_review

    	count += 1
    	sys.stdout.write("\r" + "%d reviews processed" % count)
    	sys.stdout.flush()

    print("")
    count = 1
    for key_val in sentimentDict:
    	sys.stdout.write("\r" + "%d reviews inserted" % count)
    	sys.stdout.flush()
    	count += 1

    print("")
    print("Reviewing sentiments done!")


def load_review_data():
    for r in review.values:
        #r[1] = ' '.join([word for word in r[1].split() if word not in stopwords.words("english")])

        if r[0] in business_dict.keys():
            business_obj = business_dict[r[0]]
            business_obj.review_list.append(r[1])
            business_obj.tip_list.append(r[2])
        else:
            business = BusinessReview()
            business.business_id = r[0]
            business.review_list.append(r[1])
            business.tip_list.append(r[2])
            business_dict[r[0]] = business

    print(len(business_dict))


business_dict = {}
review = pd.read_pickle('Data/review.pkl')
tips = pd.read_pickle('Data/tip.pkl')
sentimentDict = {}

load_review_data()
print('data loaded...')
'''
Splitting the data
'''
#train_review, test_review = train_test_split(review["review"], review["categories"], test_size=0.20, random_state=4212)
#print('Review Data loaded...')

#train_tip, test_tip,  = train_test_split(tips["tip"], tips["categories"], test_size=0.20, random_state=4212)
#print('Tips Data loaded...')

for business in business_dict:
    b = business_dict[business]
    analyze_reviews(b.review_list)


