import gensim
import sys
import nltk
import pickle
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

from textblob import TextBlob
from string import punctuation

def analyzeSentiment(inputStrings):
    positive = 0
    negative = 0
    neutral = 0

    positiveReviews = []
    negativeReviews = []
    neutralReviews = []

    print("Analyzing sentiments")
    count = 1
    for string in inputStrings:
        text = TextBlob(string)

        if text.sentiment.polarity < -0.15:
            negative += 1
            # negativeReviews.append(string)
        elif text.sentiment.polarity > 0.15:
            positive += 1
            positiveReviews.append(string)
            # print text.noun_phrases

        else:
            neutral += 1
            # neutralReviews.append(string)

        sys.stdout.write("\r" + "%d sentiments analyzed" % count)
        sys.stdout.flush()
        count += 1

    print("")
    print("Positive review - %d" % positive)
    print("Negative review - %d" % negative)
    print("Neutral review - %d" % neutral)

    # for string in negativeReviews:
    #     print string
    #     dummy = raw_input("")

    buildTopicModel(positiveReviews)

def getWordVector(inputString):
    tokenizer = RegexpTokenizer(r'\w+\'?\w+')

    # default English stop words list
    en_stop = set(stopwords.words('english'))

    # Create p_stemmer of class PorterStemmer
    # It is considered to be the best for finding word roots
    p_stemmer = PorterStemmer()

    raw = inputString.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # now POS words which are nouns, adjectives, adverbs and verbs
    pos_tagged = nltk.pos_tag(stopped_tokens)

    # stem tokens
    # p_stemmer.stem(i[0]) and other additions in if condition - or i[1][0] == 'R' or i[1][0] == 'V'

    stemmed_tokens = [i[0]
                        for i in pos_tagged
                        if i[1][0] == 'N'] # or i[1][0] == 'J']

    return stemmed_tokens

def getTopics(bagOfWords):
    topics = pd.read_pickle("trained-model-topics")
    ldaModel = pd.read_pickle("trained-negative-model")
    ldaDictionary = pd.read_pickle("model-dictionary")

    topics_found = []
    models = ldaModel[ldaDictionary.doc2bow(bagOfWords)]
    models = sorted(models, key=lambda k: k[1], reverse = True)

    for single_topic in models:
        topics_found.append(topics[single_topic[0]])
        if len(topics_found) > 2:
            break

    return topics_found

def buildTopicModel(inputStrings):


    # list for tokenized documents in loop
    texts = []

    count = 1
    # loop through document list
    for i in inputStrings:
        # clean and tokenize document string
        # add tokens to list
        texts.append(getWordVector(i))

        sys.stdout.write("\r" + "%d strings processed" % count)
        sys.stdout.flush()
        count += 1

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    print
    print("Bag of words is %d documents long" % len(corpus))

    # generate LDA model
    ldaModel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=10, id2word=dictionary, workers=2, passes=1)

    print('Modelling done for LDA')
    for topic in ldaModel.print_topics(num_topics=10, num_words=10):
        print(topic)

    pickle.dump(ldaModel, open("trained-negative-model",'w'))
    pickle.dump(dictionary, open("model-dictionary", 'w'))
    print("Model written to file --trained-negative-model--")