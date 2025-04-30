import numpy as np;
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log(p) for p in ps if p>0])

def information_gain(y, X_column, threshhold):
    #parent entropy
    parent_entropy = entropy(y)

    left_node, right_node = split(X_column, threshhold)

def split(X, thr):
    pass

def most_common_value(y):
    counter = Counter(y)
    value = counter.most_common(1)[0][0]
    return value

def best_split(X, y):
    thr = None
    gain = information_gain(y, X, thr)

def build_tree():
    pass

def predict():
    pass