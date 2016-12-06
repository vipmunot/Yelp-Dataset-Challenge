
import sys
import pymongo
import TopicModeller as tm
import pickle

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def main():
    try:
        c = MongoClient('localhost', 27017)
    except ConnectionFailure, e:
        print "Could not connect to MongoDB: %s" % e

