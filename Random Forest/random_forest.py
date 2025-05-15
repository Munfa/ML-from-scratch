import numpy as np

def entropy(y):
    count = np.bincount(y)
    probabilities = count/len(y)
    for p in probabilities:
        if p>0:
            entropy = -p * np.log2(p)
    return entropy

def information_gain(y, y_left, y_right):
    parent_entropy = entropy(y)
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)
    e_left = entropy(y_left)
    e_right = entropy(y_right)

    child_entropy = (n_left/n) * e_left + (n_right/n) * e_right
    gain = parent_entropy - child_entropy

    return gain

