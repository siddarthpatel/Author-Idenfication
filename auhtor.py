import os
import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
train_str = "/Users/siddarthpatel/Desktop/author/C50/C50train"
test_str = "/Users/siddarthpatel/Desktop/author/C50/C50test"
train_path = os.listdir(train_str)
test_path = os.listdir(test_str)

def tokenize_doc(doc):

    bow = defaultdict(int)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1
    return dict(bow)


def get_dictionary(path):
    auth_to_label = {}
    label_to_auth = {}
    i = 0
    for auth_name in path:
        if not auth_name.startswith('.'):
            auth_to_label[auth_name] = i
            label_to_auth[i] = auth_name
            i+=1
    return (dict(auth_to_label), dict(label_to_auth))


def get_bows_labels(path, string):
    x = []
    y = []
    auth_to_label, label_to_auth = get_dictionary(path)
    for auth_name in path:
        if not auth_name.startswith('.'):
            for doc_name in os.listdir(string + '/' + auth_name):
                path_to_doc = string + '/' + auth_name + '/' + doc_name
                with open(path_to_doc, 'r') as doc:
                    content = doc.read()
                    x.append(content)
                    y.append(auth_to_label.get(auth_name))
    return (x, y)

(x_train, y_train) = get_bows_labels(train_path, train_str)
x_train_vec = vectorizer.fit_transform(x_train).toarray()
y_train_vec = np.asarray(y_train)

(x_test, y_test) = get_bows_labels(test_path, test_str)
x_test_vec = vectorizer.transform(x_test).toarray()
y_test_vec = np.asarray(y_test)

model = GaussianNB()

model.fit(x_train_vec, y_train_vec)

y_pred = model.predict(x_test_vec)

print(metrics.accuracy_score(y_test_vec, y_pred))
