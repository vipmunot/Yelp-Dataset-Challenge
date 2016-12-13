from pymongo import MongoClient
import sys
import naivebayes as bayes
import preprocess as pre
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
from math import ceil
import operator

def ismatch(predicted, truth):
	predicted = [(x, predicted[x]) for x in range(len(predicted))]
	predicted = sorted(predicted, key=operator.itemgetter(1), reverse=True)
	truth = [(x, truth[x]) for x in range(len(truth))]
	truth = [y for y in sorted(truth, key=operator.itemgetter(1), reverse=True) if y[1]]

	count = 0
	for l in predicted[:7]:
		for t in truth:
			if l[0] == t[0]:
				count += 1

	return True if (count > int(len(truth) * 0.45)) or (len(truth) < 1) else False


split = 15000
	ct_train_data = TfidfVectorizer()
	ct_train_labels = CountVectorizer(tokenizer=lambda t: t.split("|"))
	X = ct_train_data.fit_transform(open(pre.data_file, 'r'))
	print "Train passages vectorized"
	Y = ct_train_labels.fit_transform(open(pre.label_file, 'r'))	
	Y = Y.todense()
	print "Train labels vectorized"
	if not os.path.exists(randomforest_model_file):
		rfModel = OneVsRestClassifier(LogisticRegression())
		rfModel.fit(X[:split], Y[:split])
	else:
		rfModel = pickle.load(open(randomforest_model_file, "rb"))

	print "Performing prediction"

	predicted = rfModel.predict_proba(X[split:])
	correct = 0
	count = 0
	for i in range(len(predicted)):
		if ismatch(predicted[i].tolist(), Y[split+i].tolist()[0]):
			correct += 1		
		count += 1
		sys.stdout.write("\r" + "%d businesses' review data scored with %.2f accuracy so far .." % (count, (correct * 100.0 / count)					
		sys.stdout.flush()

