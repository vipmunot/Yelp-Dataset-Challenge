from collections import defaultdict
import operator
import re
from math import log
import random
import pandas
import re
import os

filename = 'distinctive_words.txt'

"""
This function prints the top 10 words associated with each of the 20 topics into the file "distinctive_words.txt".
It is used only in the part2 of this assignment. Input is

all_words - the word count data structure created by train() function
"""
def top10(all_words):
	if os.path.exists(filename):
		os.remove(filename)

	output = open(filename, 'w')
	for topic in all_words:
		output.write('-'*50 + "\n")
		output.write("Top 10 words for " + topic + "\n")
		output.write('-'*50 + "\n")
		top_words = sorted(((k, v) for k,v in all_words[topic].iteritems()), key=lambda tup: tup[1], reverse=True)
		for word in top_words[:10]:
			output.write(str(word[0]) + " - " + str(int(word[1])) + "\n")

	output.close()
	print "Output written to file", filename


def defaultvalue():
	return 0.0000000001

def defaultcollection():
	return defaultdict(defaultvalue)

model = {}
all_words = defaultdict(defaultcollection)
topic_likelihood = defaultdict(float)		
topicwise_counts = defaultdict(float)
topics = []

"""
Function to train the Bayes model. This function is common for the part1 (spam, non-spam) and part2 (topics) case.
For part2 it implements the EM algorithm when the fraction is < 1. The number of iterations for EM are kept to 3 through experimental results for
different values on this dataset. The idea of EM algorithm is pretty simple
E step - assumes the topic assignment for each document is fixed and then calculates the word likelihood ie. P(W|T) over all topics
M step - assumes the word likelihoods are fixed and then calculates the posterior i.e P(T|w1 w2 .. wn) where w1 .. wn belong to a document D.
This is done for all documents D in the train dataset.

The input arguments for it are in the form of 

data_row - this is the data point for which we train in an online fashion
fraction - is the fraction input by the user. This is applicable only for part2 and for part1 it takes on its default assignment of 1
part - for which part of the assignment is the function being called viz. "part1" or "part2"
featuretype - can take values of "binary" or "frequency". It mainly controls how the feature vectors are implemented
"""
def train_online(data_row):
	topics = data_row['labels']	
	
	for t in topics:			
		for word in data_row['text']:				
			topic_likelihood[t] += 1
			all_words[t][word] += 1
			topicwise_counts[t] += 1
	
def finalize_model():
	global topic_likelihood
	global all_words
	global topicwise_counts

	total = sum(topic_likelihood.itervalues(), 0.0)
	topic_likelihood = {k: v / total for k, v in topic_likelihood.iteritems()}
	top10(all_words)

	# final structure contained in the dictionary to be pickled
	model['words'] = all_words	
	model['topic_likelihood'] = topic_likelihood
	model['topicwise_counts'] = topicwise_counts
	model['topics'] = all_words.keys()
	return model

def partial_match(predicted, truth):
	count = 0.0
	for l in predicted:
		for t in truth:
			if l == t:
				count += 1

	return count


def assign_model(model):
	global topic_likelihood
	global all_words
	global topicwise_counts
	global topics

	topic_likelihood = model['topic_likelihood']
	topics = model['topics']
	all_words = model['words']
	topicwise_counts = model['topicwise_counts']

"""
This function tests the trained Naive Bayes model on the data contained in test_dir.

model - the trained model returned by the train() function
test_dir - directory which contains test documents organized into directories corresponding to actual topic names
"""
# confusion_matrix = [[0]*len(topics) for i in range(len(topics))]
# average = 0.0
# average_total = 0.0

def test_online(test_row):
	global topic_likelihood
	global all_words
	global topicwise_counts
	global topics

	topic_assignment = defaultdict(float)

	# print sum(len(v) for k,v in all_words.iteritems()), "words present in the vocabulary"	
	for word in test_row['text']:
		for t in topics:
			topic_assignment[t] += log((all_words[t][word] / topicwise_counts[t]), 2)

	for t in topics:
		topic_assignment[t] += log(topic_likelihood[t], 2)
			
	predicted_topics = sorted(topic_assignment.iteritems(), key=operator.itemgetter(1), reverse=True)
	predicted_topics = [x[0] for x in predicted_topics[:7]]
	test_row['predicted_labels'] = predicted_topics

	del topic_assignment
	del pre
	# print [x[0] for x in predicted_topics[:3]], "vs", test_row['labels']	
	# return partial_match(predicted_topics, test_row['labels'])

	# if predicted_topic == t:
	# 	correct += 1
	# confusion_matrix[topics.index(t)][topics.index(predicted_topic)] += 1

	# average_total += total
	# average += correct
	# print "Accuracy for", t, "is", str(int(correct)) + "/" + str(int(total)), str(round(((correct / total) * 100), 2)) + "%"

	# print "Average accuracy is", str(round(((average / average_total) * 100), 2)) + "%"

	# print "\n", "Here is the confusion matrix"
	# matrix = pandas.DataFrame(confusion_matrix, topics, topics)
	# print matrix