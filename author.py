import os
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

train_str = "/Users/siddarthpatel/Desktop/author/C50/C50train"
test_str = "/Users/siddarthpatel/Desktop/author/C50/C50test"
train_path = os.listdir(train_str)
test_path = os.listdir(test_str)
words = defaultdict(int)

def tokenize_doc(doc):

    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
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
                    tokens = tokenize_doc(content)
                    for word in tokens:
                        words[word] += tokens[word]
                    x.append(content)
                    y.append(auth_to_label.get(auth_name))
    return (x, y)


def word_frequency_graph():
    get_bows_labels(train_path, train_str)
    get_bows_labels(test_path, test_str)
    x = []
    y = []
    X_LABEL = "log(rank)"
    Y_LABEL = "log(frequency)"
    frequency = sorted(words.values(), reverse=True)
    ranks = range(1, len(words)+1)
    x = [math.log(r) for r in ranks]
    y = [math.log(f) for f in frequency]
    plt.scatter(x, y)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.savefig('word_frequency_graph.png')


# word_frequency_graph()




