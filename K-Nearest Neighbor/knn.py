import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

########## Calculating the distance between the points ############
def eucleadian_distance(x1, x2):
    d = np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
    return d

########## Predicting the category for the new point ###########
def predict(x_train, y_train, new_point, k):
    predictions = []
    for x1 in new_point:
        distances = [eucleadian_distance(x1, x2) for x2 in x_train]
        sorted = np.argsort(distances)[:k]       ###### getting k number of indices of sorted distances ######
        nearest_target = [y_train[i] for i in sorted]   ###### getting the target of nearest points ######
        most_common = Counter(nearest_target).most_common()[0][0]
        predictions.append(most_common)
        
    return predictions

########## Loading Iris dataset and setting X and y values ###########
df = datasets.load_iris()
X, y = df.data, df.target

########## Splitting the dataset into train and test set ###########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

pred = predict(X_train, y_train, X_test, k=5)
########## Calculating accuracy ###########
accuracy = np.sum(pred == y_test) / len(y_test)
print(float(accuracy))

plt.scatter(X_train[:,1], X_train[:,3], c=y_train, cmap=ListedColormap(['b', 'g', 'r']), s=20)
plt.scatter(X_test[:,2], X_test[:,3], c=pred, cmap=ListedColormap(['b', 'g', 'r']), marker='*')
plt.colorbar()
plt.show()
