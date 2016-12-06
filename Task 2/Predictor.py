import sys
import pymongo
import TopicModeller as tm
import pickle

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

topics_found_positive = []
topics_found_negative = []


def main():
    try:
        c = MongoClient('localhost', 27017)
    except ConnectionFailure, e:
        print "Could not connect to MongoDB: %s" % e

    # getting a handle to the database
    db = c["Yelp"]
    print "Businesses database connected successfully"

    positive_topics = ["Customer Service",
                 "Food",
                 "Bar & Liquor",
                 "Overall Quality",
                 "Mexican Food",
                 "Breakfast",
                 "Ambiance & Hospitality",
                 "Expensive",
                 "Location",
                 "Enterntainment"]
    pickle.dump(positive_topics, open("trained-model-positive-topics",'w'))

    negative_topics = ["Mexican Food",
                 "Staff and Service",
                 "Food",
                 "Coffee and Cakes ",
                 "Ambiance & Hospitality",
                 "Bad Service",
                 "Not Pet Friendly",
                 "Parking and Utilities",
                 "Delivery Services",
                 "Enterntainment"]
    pickle.dump(negative_topics, open("trained-model-negative-topics",'w'))

    positive_topics_data = pickle.load(file('trained-model-positive-topics'))
    negative_topics_data = pickle.load(file('trained-model-negative-topics'))
    ldaModel_positive = pickle.load(file('trained-positive-model'))
    ldaModel_negative = pickle.load(file('trained-negative-model'))
    ldaDictionaryPositive = pickle.load(file('model-dictionary-positive.txt'))
    ldaDictionaryNegative = pickle.load(file('model-dictionary-negative.txt'))


    businesses = db.reviewsSentiment.find().limit(100)
    for business in businesses:
        businesses = db.yelp_academic_dataset_.find().limit(100)
        topic_positive = {}
        topic_negative = {}


        print '\n\nBusiness Id :: ', business["business_id"]
        count_n = 1
        query = {"sentiment" : { "$lt" : -0.05 },  "business_id" : {"$eq" : business["business_id"]} }
        reviews_set_negative = []
        for review in db.reviewsSentiment.find(query):
            if count_n >= 1:
                reviews_set_negative.append(review["text"])
            count_n += 1
        print("\r" + "%d negative reviews processed" % count_n)


        reviews_set_positive = []
        count_p = 1
        query = {"sentiment" : { "$gt" : -0.05 } ,  "business_id" : {"$eq" : business["business_id"]} }
        for review in db.reviewsSentiment.find(query):
            if count_p >= 1:
                reviews_set_positive.append(review["text"])
            count_p += 1

        print("\r" + "%d positive reviews processed" % count_p)

        for item in reviews_set_positive:
            models = ldaModel_positive[ldaDictionaryPositive.doc2bow(tm.get_word_vector(item))]
            models = sorted(models, key=lambda k: k[1], reverse = True)
            for single_topic in models:
                topics_found_positive.append(positive_topics[single_topic[0]])
                topic_positive[positive_topics_data[single_topic[0]]] = single_topic[1]
                if len(topics_found_positive) > 2:
                    break

        for item in reviews_set_negative:
            models = ldaModel_negative[ldaDictionaryNegative.doc2bow(tm.get_word_vector(item))]
            models = sorted(models, key=lambda k: k[1], reverse = True)
            for single_topic in models:
                topics_found_negative.append(negative_topics[single_topic[0]])
                topic_negative[negative_topics_data[single_topic[0]]] = single_topic[1]
                if len(topics_found_negative) > 2:
                    break

        sorted(topic_positive, key=topic_positive.__getitem__, reverse=True)
        sorted(topic_negative, key=topic_negative.__getitem__, reverse=True)

        positive = topic_positive.keys()
        negative = topic_negative.keys()

        print 'POSITIVE :: '
        for top in positive:
            print top, ' --> ',topic_positive[top],'  '

        print 'NEGATIVE :: '
        for top in negative:
            print top, ' --> ',topic_negative[top],'  '

    c.close()

if __name__ == "__main__":
    main()
