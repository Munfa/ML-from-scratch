import numpy as np;
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def gini_impurity(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts/counts.sum()
    g_impurity = 1 - np.sum(probabilities ** 2)
    return g_impurity

def split(X, feat_idx, thr):
    left_split = X[:, feat_idx] <= thr
    right_split = X[:, feat_idx] > thr

    return left_split, right_split
    
def classify(y):
    unique_classes, class_count = np.unique(y, return_counts=True)
    index = class_count.argmax()
    classification = unique_classes[index]

    return classification

def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = 1 #Maximum possible gini

    n_features = X.shape[1]

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for thr in thresholds:
            # create children node
            left_node, right_node = split(X, feature_idx, thr)
            y_left, y_right = y[left_node], y[right_node]

            if len(left_node) == 0 or len(right_node) == 0:
                continue

            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            total_gini = (len(y_left) * gini_left + len(y_right) * gini_right)/len(y)

            if total_gini < best_gini:
                best_gini = total_gini
                best_feature = feature_idx
                best_threshold = thr

    return best_feature, best_threshold 

def build_tree(X, y, depth=0, max_depth=5, min_samples=50):
    n_samples = X.shape[0]
    if len(np.unique(y)) == 1 or depth == max_depth or n_samples<min_samples:
        return classify(y)
    
    best_feature, best_threshold = best_split(X, y)

    left_idxs, right_idxs = split(X, best_feature, best_threshold)

    left_tree = build_tree(X[left_idxs, :], y[left_idxs], depth+1, max_depth, min_samples)
    right_tree = build_tree(X[right_idxs, :], y[right_idxs], depth+1, max_depth, min_samples)
    
    return best_feature, best_threshold, left_tree, right_tree


def traverse_tree(tree, x):
    if not isinstance(tree, tuple):
        return tree

    feature, thr, left, right = tree

    if x[feature] <= thr:
        return traverse_tree(left, x)
    else:
        return traverse_tree(right, x)

def predict(tree, X):
    pred = np.array([traverse_tree(tree, x) for x in X])
    return pred

df = datasets.load_iris()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

tree = build_tree(X_train, y_train)

y_pred = predict(tree, X_test )
accuracy = np.mean(y_pred == y_test)

print(accuracy)

plt.scatter(range(len(y_test)), y_test, label="Actual", marker='o', s=80, color='blue')
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='x', color='red')
plt.title("Decision Tree")
plt.legend(loc='upper right')
plt.show()
