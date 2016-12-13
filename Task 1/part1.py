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

bayes_model_file = 'bayes-model'
neuralnet_model_file = 'neural-net-model'
randomforest_model_file = 'random-forest-model'

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

def bayes_model(mode="train", trained_model=None):
	client = MongoClient('mongodb://localhost:27017/')
	db = client['local']	

	if mode == "train":
		if os.path.exists(bayes_model_file):
			os.remove(bayes_model_file)

		count = 0
		for item in db.train_part1.find():
			bayes.train_online(item)

			count += 1
			sys.stdout.write("\r" + "%d businesses' review data trained" % count)
			sys.stdout.flush()

		print "\n" + "Model trained!"
		trained_model = bayes.finalize_model()
		pickle.dump(trained_model, open(bayes_model_file, "wb"))
		print "Bayes Model written to", bayes_model_file

	elif mode == "test":
		bayes.assign_model(trained_model)
		
		count = 0
		for item in db.test_part1.find().limit(14000):
			if "predicted_labels" not in item:
				bayes.test_online(item)
				db.test_part1.update({"_id": item['_id']}, item)

			count += 1
			sys.stdout.write("\r" + "%d businesses' review data predicted so far .." % (count))
			sys.stdout.flush()

	elif mode == "score":
		print "Scoring the labels"
		correct = 0
		count = 0
		hamming_score = []
		precision = []
		recall = []

		for item in db.test_part1.find({"predicted_labels" : { "$exists" : True, "$ne" : None }}):
			union_set = set()
			for l in item['predicted_labels']:
				union_set.add(l)
			for l in item['labels']:
				union_set.add(l)

			# print union_set

			match_count = bayes.partial_match(item['predicted_labels'], item['labels'])		
			if (match_count > int(len(item['labels']) * 0.45)) or (len(item['labels']) < 1):
				correct += 1

			if len(item['labels']) > 0:
				# print (match_count / len(union_set)),
				hamming_score.append((match_count / len(union_set)))
				precision.append((match_count / len(item['predicted_labels'])))
				recall.append((match_count / len(item['labels'])))

			count += 1
			sys.stdout.write("\r" + "%d businesses' review data scored with %.2f accuracy so far .." % (count, (correct * 100.0 / count)))																																																																																																																																									
			sys.stdout.flush()
				
		print ""
		print "Avg precision", sum(precision) / len(precision)
		print "Avg recall", sum(recall) / len(recall)
		print "Avg hamming score", sum(hamming_score) / len(hamming_score)
		print "Correctly predicted %d/%d businesses" % (correct, count)
		print "Accuracy is %.2f" % ((correct * 100.0) / count)

	print ""
	client.close()

def neural_net(mode="train"):
	if mode == "train":
		ct_train_data = CountVectorizer()
		ct_train_labels = CountVectorizer(tokenizer=lambda t: t.split("|"))
		ct_train_data = ct_train_data.fit(open(pre.data_file, 'r'))
		ct_train_labels = ct_train_labels.fit(open(pre.label_file, 'r'))
		X_train_counts = ct_train_data.transform(open(pre.train_data_file, 'r'))
		print "Train passages vectorized"
		Y_train_labels = ct_train_labels.transform(open(pre.train_label_file, 'r'))
		print "Train labels vectorized"

		mlp = MLPClassifier(hidden_layer_sizes=(100,), batch_size=100 ,activation='logistic', max_iter=25, alpha=1e-4, solver='sgd', verbose=True, \
		tol=0.0001, random_state=1, learning_rate_init=.01)
		mlp.fit(X_train_counts, Y_train_labels) #train the network
		print "\nNetwork trained!"
		pickle.dump(mlp, open(neuralnet_model_file, "wb"))
		print "Neural Net Model written to", neuralnet_model_file
	elif mode == "test":
		print "Model is already trained! Reading the file ..."
		mlp = pickle.load(open(neuralnet_model_file, "rb"))
		ct_test_data = CountVectorizer()
		ct_test_labels = CountVectorizer(tokenizer=lambda t: t.split("|"))
		ct_test_data = ct_test_data.fit(open(pre.data_file, 'r'))
		ct_test_labels = ct_test_labels.fit(open(pre.label_file, 'r'))
		X_test_counts = ct_test_data.transform(open(pre.test_data_file, 'r'))
		print "Test passages vectorized"
		Y_test_labels = ct_test_labels.transform(open(pre.test_label_file, 'r'))
		print "Test labels vectorized"
		print("\nTest set score: %f" % mlp.score(X_test_counts, Y_test_labels))


if sys.argv[1] == "naive-bayes":
	mode = str(sys.argv[2])
	if mode == "train":
		print "Training the bayes model"
		bayes_model(mode="train")
	elif mode == "test":
		if os.path.exists(bayes_model_file):
			print "Model is already trained! Reading the file ..."
			trained_model = pickle.load(open(bayes_model_file, "rb"))
			bayes_model(mode="test", trained_model=trained_model)
		else:
			print "Model file not found :("
	elif mode == "score":
			bayes_model(mode="score")
	elif mode == "join":	
		pre.join_per_business()
	elif mode == "split":
		pre.split()	
elif sys.argv[1] == 'neural-net':
	mode = str(sys.argv[2])
	if mode in ["train", "test"]:
		neural_net(mode)
		# if os.path.exists(bayes_model_file):
		# 	print "Model is already trained! Reading the file ..."
		# 	trained_model = pickle.load(open(bayes_model_file, "rb"))
		# 	bayes_model(mode="test", trained_model=trained_model)
		# else:
		# 	print "Model file not found :("
	elif mode == "score":
		bayes_model(mode="score")
	elif mode in ["join", "split"]:	
		pre.join_and_split(mode=mode)
elif sys.argv[1] == 'scikit-classifier':
	split = 15000
	ct_train_data = TfidfVectorizer()
	ct_train_labels = CountVectorizer(tokenizer=lambda t: t.split("|"))
	X = ct_train_data.fit_transform(open(pre.data_file, 'r'))
	print "Train passages vectorized"
	Y = ct_train_labels.fit_transform(open(pre.label_file, 'r'))	
	Y = Y.todense()
	print "Train labels vectorized"
	if not os.path.exists(randomforest_model_file):
		# svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
		# 	multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=100)
		# rfModel = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1))
		rfModel = OneVsRestClassifier(RandomForestClassifier(n_estimators=25, n_jobs=4, verbose=1, max_depth=5), n_jobs=2)
		# rfModel = OneVsRestClassifier(LogisticRegression())
		# rfModel = OneVsRestClassifier(LinearSVC())
		# rfModel = KNeighborsClassifier(n_neighbors=10, n_jobs=4)
		# rfModel = MLPClassifier(hidden_layer_sizes=(100,), batch_size=100 ,activation='logistic', max_iter=25, alpha=1e-4, solver='sgd', verbose=True, \
		# tol=0.0001, random_state=1, learning_rate_init=.1)
		rfModel.fit(X[:split], Y[:split])
		# pickle.dump(rfModel, open(randomforest_model_file, "wb"))
	else:
		rfModel = pickle.load(open(randomforest_model_file, "rb"))

	print "Performing prediction"

	predicted = rfModel.predict_proba(X[split:])
	correct = 0
	count = 0
	for i in range(len(predicted)):
		# print predicted[i].tolist()
		# print Y[10000+i].tolist()
		# raw_input()
		if ismatch(predicted[i].tolist(), Y[split+i].tolist()[0]):
			correct += 1		
		# if sum([int(ceil(x)) & y for (x,y) in zip(predicted[i].tolist(), Y[split+i].tolist()[0])]) > 1:
		# 	correct += 1
		count += 1
		sys.stdout.write("\r" + "%d businesses' review data scored with %.2f accuracy so far .." % (count, (correct * 100.0 / count)))																																																																																																																																									
		sys.stdout.flush()
			
	print ""
	print "Correctly predicted %d/%d businesses" % (correct, count)
	print "Accuracy is %.2f" % ((correct * 100.0) / count)

	# scores = cross_val_score(svc, X, Y, cv=1)
	# print "Accuracy-->", scores.mean()
	# print scores
else:
	print "Unknown argument"
