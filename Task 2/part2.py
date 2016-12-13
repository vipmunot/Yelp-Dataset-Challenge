from textblob import TextBlob
from pymongo import MongoClient
import sys

sentiment_threshold = 0.2
def extract_pros_cons(business_id):
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']
	
	positive = set()
	negative = set()
	for review in db.review.find({'business_id' : business_id}):
		blob = TextBlob(review['text'])
		for ph in blob.noun_phrases:
			if len(ph.split()) > 1 and len(ph.split()) < 4:
				if blob.sentiment[0] < -sentiment_threshold:
					negative.add((ph, blob.sentiment[0]))
				if blob.sentiment[0] > sentiment_threshold:
					positive.add((ph, blob.sentiment[0]))

	for tip in db.tip.find({'business_id' : business_id}):
		blob = TextBlob(tip['text'])
		for ph in blob.noun_phrases:
			if len(ph.split()) > 1 and len(ph.split()) < 4:
				if blob.sentiment[0] < -sentiment_threshold:
					negative.add((ph, blob.sentiment[0]))
				if blob.sentiment[0] > sentiment_threshold:
					positive.add((ph, blob.sentiment[0]))

	positive = sorted(positive, key=lambda x: x[1], reverse=True)
	negative = sorted(negative, key=lambda x: x[1], reverse=True)
	return [x[0] for x in positive[:5]], [x[0] for x in negative[-5:]]

def process():
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']
	print "Items to be read are", db.business.count()

	count = 0
	for business in db.business.find().skip(2000).limit(2000):		
		pros, cons = extract_pros_cons(business['business_id'])
		if pros:
			business['pros'] = pros
		if cons:
		 	business['cons'] = cons
		db.tagged_businesses.insert(business)
		# data.write(cleaned_data['text'] + "\t" + cleaned_data['labels'] + "\n")
		count += 1
		sys.stdout.write("\r" + "%d businesses' review data processed" % count)
		sys.stdout.flush()

	print ""
	client.close()

process()