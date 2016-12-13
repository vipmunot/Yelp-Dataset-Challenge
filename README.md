# Yelp Dataset Challenge
We have used the dataset released by Yelp to apply various machine learning and text mining techniques. The dataset contains a wealth of information in the form of reviews and tips. These are opinions expressed by users for over 82K businesses.

ILS Z534 Search Project
---
Team Members: <br/>
1. Vipul Munot <br/>
2. Pralhad Sapre <br/>
3. Rutuja Kulkarni <br/>
4. Neelam Tikone <br/>
5. Nishant Salvi <br/>

The code for this project is divided into two tasks

Task 1
---
Deals with the prediction of labels for a given business from its reviews. We take a supervised learning approach to this problem. We split the dataset in an 80-20 fashion and apply algorithms like Naive Bayes, Random Forests and Logistic Regression on the word vectors of reviews. The aspect which makes this problem challenging is the multi-label multi-class output. We tackle this problem by using a partial accuracy measure where 40% match with the ground truth is considered as a match.

The main code for this part is contained in the file part1.py. Other files are modules which aid in preprocessing and model building.

Task 2
---
Here we try to find the pros and cons of a restaurant.

The first approach for this problem relies on using Topic Modelling. The main idea here is to pluck out the nouns and adjectives from the reviews and build two models out of it. All the positive reviews help in building the pros model and the negative reviews help to build the cons model. Once this modelling is done we assign labels to word clusters which become the pros and cons label for any review tested on the two models.

The main code file doing the predictions is Predictor.py

The second approach is a really simple noun phrase extraction approach. We only rely on bi-grams and tri-grams as indicators of pros and cons. The top three noun phrases from positive reviews are assigned as pros and top three phrases from negative reviews become the cons for a restaurant.

The main code file for the 2nd approach is part2.py.
