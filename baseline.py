import os
import numpy as np
from author import get_bows_labels, get_dictionary
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
train_str = "/Users/siddarthpatel/Desktop/author/C50/C50train"
test_str = "/Users/siddarthpatel/Desktop/author/C50/C50test"
train_path = os.listdir(train_str)
test_path = os.listdir(test_str)

def get_vectors():

	(x_train, y_train) = get_bows_labels(train_path, train_str)
	x_train_vec = vectorizer.fit_transform(x_train).toarray()
	y_train_vec = np.asarray(y_train)

	(x_test, y_test) = get_bows_labels()
	x_test_vec = vectorizer.transform(x_test).toarray()
	y_test_vec = np.asarray(y_test)

	return (x_train_vec, y_train_vec, x_test_vec, y_test_vec)

def train_predict():

	(x_train, y_train, x_test, y_test) = get_vectors()
	model = GaussianNB()
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	return (y_test, y_pred)

def model_accuracy():

	(y_test, y_pred) = train_predict()
	print("The baseline accuracy is:")
	print(metrics.accuracy_score(y_test, y_pred) * 100)

model_accuracy()