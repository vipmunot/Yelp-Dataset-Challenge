from textblob import TextBlob
import sys
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords


def main():
    c = MongoClient('localhost', 27017)

    # getting a handle to the database
    db = c["Yelp"]
    print "Database connected successfully"

    sentimentDict = {}

    count = 1
    review_cursor = db.reviews.find()
    f_positive = open('positive.txt','w')
    f_negative = open('negative.txt','w')

    for one_review in review_cursor:
    	if True:
            review_text = ' '.join([word for word in one_review["text"].split() if word not in stopwords.words("english")]).encode('utf-8').strip()
            #tokens = nltk.word_tokenize(review_text)
            one_review["sentiment"] = TextBlob(one_review["text"]).sentiment.polarity
            sentimentDict[one_review["review_id"]] = one_review
            if one_review["sentiment"] > 0.05:
                f_positive.write(review_text+'\n')
            else:
                f_negative.write(review_text+'\n')


    	count += 1
    	if count % 10000 == 0:
    	    sys.stdout.write("\r" + "%d reviews processed" % count)
    	    sys.stdout.flush()


    f_positive.close()
    f_negative.close()
    print ""
    count = 1
    for key_val in sentimentDict:
    	db.reviewsSentiment.insert(sentimentDict[key_val])

    	sys.stdout.write("\r" + "%d reviews inserted" % count)
    	sys.stdout.flush()
    	count += 1

    print ""
    print "Reviewing sentiments done!"

    c.close()

if __name__ == "__main__":
    main()