import numpy as np;
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def gini_impurity(y):
    '''
        Gini impurity is the probability of misclassifying a randomly chosen element in a set
        It ranges from 0 to ~1 where 0 = purest and 1 = most impure
        Gini = 0.5 is the even split between two classes where 0.5 becomes the most impure for binary class
        A node being pure means there is no other class in that node and impure means the opposite
    '''
    
    _, counts = np.unique(y, return_counts=True) # Counting how many times each class occured
    probabilities = counts/counts.sum()          # Probability of each class
    g_impurity = 1 - np.sum(probabilities ** 2)  # gini_impurity = 1 - sum((probability_of_each_class)^2)
    return g_impurity

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

def best_split(X, y):
    '''
        Returns the best_feature and threshold to split the dataset
        The goal is to achieve lowest Gini Impurity and classify data
    '''
    best_feature = None
    best_threshold = None
    best_gini = 1 # Maximum possible impurity

    n_features = X.shape[1] # Storing number of features

    # Loop over all the features
    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx]) # Find unique values to test as thresholds
        for thr in thresholds:
            # create children node
            left_node, right_node = split(X, feature_idx, thr) # Split left and right indices for the nodes based on the threshold
            y_left, y_right = y[left_node], y[right_node]      # Split class values based on the indices

            # Skip if there are no classes
            if len(left_node) == 0 or len(right_node) == 0:
                continue

            gini_left = gini_impurity(y_left) # Calculate Gini for left node
            gini_right = gini_impurity(y_right) # Calculate Gini for right node

            '''
                node weight = (total number of classes on one node)/(total number on both nodes)
                total gini = (left_node_weight) * left_gini + (right_node_weight) * right_gini

                In mathematics, 
                we can write (a*b)/c + (d*e)/c 
                as ((a*b) + (d*e))/c
            '''
            total_gini = (len(y_left) * gini_left + len(y_right) * gini_right)/len(y)

            if total_gini < best_gini: 
                best_gini = total_gini      # The best gini is the lesser gini value
                best_feature = feature_idx  # The best feature and best threshold updated based on the less gini value
                best_threshold = thr

    return best_feature, best_threshold 

def build_tree(X, y, depth=0, max_depth=5, min_samples=50):
    n_samples = X.shape[0]  # Storing number of samples
    if len(np.unique(y)) == 1 or depth == max_depth or n_samples<min_samples:
        return classify(y)  # If conditions met then the most common class becomes the label
    
    best_feature, best_threshold = best_split(X, y)     # Find the best feature and threshold

    left_idxs, right_idxs = split(X, best_feature, best_threshold)  # Returns indexes for the best split

    '''
        This recursive function keeps building the tree and increasing depth by 1 until conditions met
        The conditions are,
        1. The node is pure meaning there is only one class
        2. depth = max_depth
        3. n_samples is less than min_samples
    '''
    left_tree = build_tree(X[left_idxs, :], y[left_idxs], depth+1, max_depth, min_samples)
    right_tree = build_tree(X[right_idxs, :], y[right_idxs], depth+1, max_depth, min_samples)
    
    return best_feature, best_threshold, left_tree, right_tree


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

def predict(tree, X):
    pred = np.array([traverse_tree(tree, x) for x in X]) # Each sample is sent in a loop to classify
    return pred # Returns a numpy array of predictions

df = datasets.load_iris()   #Using Iris dataset
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

tree = build_tree(X_train, y_train) # A tuple of best feature, threshold, left and right tree is returned

y_pred = predict(tree, X_test )
accuracy = np.mean(y_pred == y_test)

print(accuracy)

# Plot the predicted and actual values
plt.scatter(range(len(y_test)), y_test, label="Actual", marker='o', s=80, color='blue')
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='x', color='red')
plt.title("Decision Tree")
plt.legend(loc='upper right')
plt.show()
