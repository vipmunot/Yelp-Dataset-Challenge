from pymongo import MongoClient
import sys
import os
import unicodedata
import re
import random
import sys

data_file = "data_restaurants.txt"
label_file = "label_restaurants.txt"

train_data_file = "train_data.txt"
train_label_file = "train_label.txt"

test_data_file = "test_data.txt"
test_label_file = "test_label.txt"

"""
Uses regular expressions to search for a proper match within a noisy token. The match regex is straightforward, 
but is powerful when removing garbage characters present in a token. It discards a token completely if a search for this regex fails.
"""
def cleanword(word):
	word = word.strip().lower()
	
	m = re.search(r"\w+.*\w+", word)
	val = m.group(0) if m else None

	return val

"""
Function to read in the stopwords from the stopwords.txt file.
It returns a dictionary which is used when reading in the data and testing on the documents.
"""
def stopwords():
	stopwords = {}
	for w in open('stopwords.txt'):
		w = w.strip()
		if len(w) > 0:
			stopwords[w] = 1
	return stopwords

def preprocess(item):
	data = {}
	text = []
	labels = []

	for word in unicodedata.normalize('NFKD', item['text']).encode('ascii','ignore').split():
		word = cleanword(word)			
		if word and word not in stopwords:
			text.append(word)

	for word in item['labels']:
		labels.append(unicodedata.normalize('NFKD', word).encode('ascii','ignore'))

	data['text'] = text
	data['labels'] = labels

	# print data['text']
	# print data['labels']
	# raw_input()
	return data


def join_per_review():
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']

	pipe = [
		# // Pipeline	
			# // Stage 1
			{
				"$lookup": {
				    "from" : "business",
				    "localField" : "business_id",
				    "foreignField" : "business_id",
				    "as" : "labels"
				}
				
			},

			# // Stage 2
			{
				"$project": {
				  "text": True,
				  "review_id": True,
				  "business_id": True,
				  "labels.categories": True,
				  "_id": False
				}
			},

			# // Stage 3
			{
				"$unwind": "$labels"
			},
		]

	count = 0
	for item in db.review.aggregate(pipeline=pipe):	
		db.aggregated_full.insert(item)
		count += 1
		sys.stdout.write("\r" + "Joined %d documents" % count)
		sys.stdout.flush()

def join_per_business():
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']
	print "Items to be read are", db.business.count()

	if os.path.exists(data_file):
		os.remove(data_file)

	# data = open(data_file, "w")

	count = 0
	for business in db.business.find():
		item = {}
		item['text'] = u""
		item['labels'] = business['categories']			
		for review in db.review.find({'business_id' : business['business_id']}):		
			item['text'] += " " + review['text']

		cleaned_data = preprocess(item)
		db.preprocessed_data.insert(cleaned_data)
		# data.write(cleaned_data['text'] + "\t" + cleaned_data['labels'] + "\n")
		count += 1
		sys.stdout.write("\r" + "%d businesses' review data processed" % count)
		sys.stdout.flush()

	print ""
	client.close()

def split(partition=0.2):
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']

	db.train_part1.drop()
	db.test_part1.drop()

	unknown_labels = 0
	count = 0
	for item in db.preprocessed_data.find():
		count += 1
		sys.stdout.write("\r" + "%d records processed" % count)
		sys.stdout.flush()
		if len(item['labels']) > 0:
			db.train_part1.insert(item) if random.random() > partition else db.test_part1.insert(item)
		else:
			unknown_labels += 1
			continue

	print "\n" + "Unknown labels were", unknown_labels, "and these reviews were dropped"

	client.close()

def join_and_split(mode="join", partition=0.2):
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']

	if mode == "join":
		print "Items to be read are", db.business.find().skip(20000).limit(20000).count()
		if os.path.exists(data_file):
			os.remove(data_file)
		if os.path.exists(label_file):
			os.remove(label_file)

		data = open(data_file, "w")
		label = open(label_file, "w")

		count = 0
		reviews = 0
		for business in db.business.find().skip(20000).limit(20000):
			item = {}
			item['text'] = u""			
			item['labels'] = business['categories']
			# item['labels'].remove("Food")

			if len(item['labels']) > 0:
				for review in db.review.find({'business_id' : business['business_id']}):		
					item['text'] += " " + review['text']
					reviews += 1

				cleaned_data = preprocess(item)
				# db.preprocessed_data.insert(cleaned_data)
				data.write(" ".join(cleaned_data['text']) + "\n")
				label.write("|".join(cleaned_data['labels']) + "\n")
			count += 1
			sys.stdout.write("\r" + "%d businesses' review data processed" % count)
			sys.stdout.flush()
		print "\n", reviews, "reviews processed"

	elif mode == "split":
		if os.path.exists(train_data_file):
			os.remove(train_data_file)
		if os.path.exists(train_label_file):
			os.remove(train_label_file)

		data = open(data_file, "r")
		label = open(label_file, "r")

		train_data = open(train_data_file, "w")
		train_label = open(train_label_file, "w")
		test_data = open(test_data_file, "w")
		test_label = open(test_label_file, "w")

		count = 0
		for d in data:
			l = label.readline()

			if random.random() > partition:
				train_data.write(d)
				train_label.write(l)
			else:
				test_data.write(d)
				test_label.write(l)

			count += 1
			sys.stdout.write("\r" + "%d businesses' review data processed" % count)
			sys.stdout.flush()

	print ""
	client.close()


stopwords = stopwords()