import numpy as np

def entropy(y):
    count = np.bincount(y)
    probabilities = count/len(y)
    for p in probabilities:
        if p>0:
            entropy = -p * np.log2(p) # Calculate entropy for every posibility
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

# Split the dataset based on threshold or conditions
def split(X, feat_idx, thr):
    left_split = X[:, feat_idx] <= thr      # Values lesser than the threshold goes to the left side of the tree
    right_split = X[:, feat_idx] > thr      # Values bigger than the threshold goes to the right side of the tree

    return left_split, right_split
    
def classify(y):
    # Labele data based on the most common class
    unique_classes, class_count = np.unique(y, return_counts=True) # Returns each class and total occurences of that class
    index = class_count.argmax()            # The index of the most occured/common class
    classification = unique_classes[index]  # Assigning the majority class as classification

    return classification

def best_split(X, y, feature_indices=None):
    '''
        Returns the best_feature and threshold to split the dataset
        The goal is to achieve higher Information Gain and classify data
    '''
    best_feature = None
    best_threshold = None
    best_gain = -1 # Minimum gain

    n_features = X.shape[1] # Storing number of features
    if feature_indices is None:
        feature_indices = range(n_features)

    # Loop over all the features
    for feature_idx in feature_indices:
        thresholds = np.unique(X[:, feature_idx]) # Find unique values to test as thresholds
        for thr in thresholds:
            # create children node
            left_node, right_node = split(X, feature_idx, thr) # Split left and right indices for the nodes based on the threshold
            y_left, y_right = y[left_node], y[right_node]      # Split class values based on the indices

            # Skip if there are no classes
            if len(left_node) == 0 or len(right_node) == 0:
                continue

            gain = information_gain(y, y_left, y_right)

            if gain > best_gain: 
                best_gain = gain      # The best gain is the higher gain value
                best_feature = feature_idx  # The best feature and best threshold updated based on the less gini value
                best_threshold = thr

    return best_feature, best_threshold 

def build_tree(X, y, min_samples=10, max_features=None):
    n_samples, n_features = X.shape  # Storing number of samples
    if len(np.unique(y)) == 1 or n_samples<min_samples or max_features is None:
        return classify(y)  # If conditions met then the most common class becomes the label
    
    feature_indices = np.random.choice(n_features, max_features, replace=False)
    best_feature, best_threshold = best_split(X, y, feature_indices)     # Find the best feature and threshold

    left_idxs, right_idxs = split(X, best_feature, best_threshold)  # Returns indexes for the branches

    '''
        This recursive function keeps building the tree and increasing depth by 1 until conditions met
        The conditions are,
        1. The node is pure meaning there is only one class
        2. max_features is None
        3. n_samples is less than min_samples
    '''
    left_tree = build_tree(X[left_idxs, :], y[left_idxs], min_samples, max_features)
    right_tree = build_tree(X[right_idxs, :], y[right_idxs], min_samples, max_features)
    
    return (best_feature, best_threshold, left_tree, right_tree)

def traverse_tree(tree, x):
    # Checks if the tree is a tuple; if it's not then it's a leaf node
    if not isinstance(tree, tuple):
        return tree # Returns the leaf node

    feature, thr, left, right = tree # Stores values from the tuple

    # This recursive function traverses the whole tree to classify sample x
    if x[feature] <= thr:
        return traverse_tree(left, x)
    else:
        return traverse_tree(right, x)

def predict_tree(tree, X):
    pred = np.array([traverse_tree(tree, x) for x in X]) # Each sample is sent in a loop to classify
    return pred # Returns a numpy array of predictions

'''
    Until now the code was for creating one Decision Tree which we will need to create Random Forest
    The below code is the functions for Random Forest
'''

# Generate randomly selected samples from data. These samples are reused
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]

# Build a forest of decision trees
def build_forest(X, y, n_trees=10, min_samples=10,):
    forest = []
    n_features = X.shape[1]
    max_features = int(np.sqrt(n_features)) #Common rule: sqrt(n_features)
    
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = build_tree(X_sample, y_sample, min_samples=min_samples, max_features=max_features)
        forest.append(tree)

    return forest

def predict_forest(X, y, forest):
    # Collect predictions from all trees
    predictions = np.array([predict_tree(tree, X) for tree in forest])

    # Majority vote for each sample
    final_preds = []
    for i in range(X.shape[0]):
        votes = predictions[:,i]
        majority_vote = classify(y)
        final_preds.append(majority_vote)

    return np.array(final_preds)
    