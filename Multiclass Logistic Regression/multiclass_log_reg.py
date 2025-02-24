import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def softmax(x, w, b):
    # z = x.w + b
    z = np.dot(x, w) + b
    # softmax = exp(z) / sum(exp(z))
    soft = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True) 
    # keepdims=True is used for keeping the shape consistent
    # axis=1 is used to sum over columns for the calculation of one sample 
    # ( the probabilities of each sample sum up to 1 )
    return soft

def onehot_encoded(y, n_samples, n_classes):
    encoded = np.zeros((n_samples, n_classes)) # creating a matrix of zeroes wheres rows= n_samples col= n_classes 
    encoded[np.arange(n_samples), y] = 1 # using y as the col indices here and assigning 1 to the correct position
    return encoded

def loss_function(y, y_hat, n_samples):
    # loss = - (1/n) * y * log(y_pred)
    loss = -(1/n_samples) * np.sum(y * np.log(y_hat)) 
    return loss

def gradient_descent(x, y, w, b, n_samples, lr, epochs):
    for epoch in range(epochs):
        y_hat = softmax(x, w, b) # getting probabilites 
        loss = loss_function(y, y_hat, n_samples)
        dw = (1/n_samples) * np.dot(x.T, (y_hat - y)) # dw = (1/n) * transpose(x) * (y_pred - y)
        db = (1/n_samples) * np.sum(y_hat - y) # db = (1/n) * (y_pred - y)
        w = w - (lr * dw)
        b = b - (lr * db)
        if (epoch % 200) == 0:
            print('Epoch: ', epoch, ' Loss: ',loss) # printing cost after every 200 epoch
    return w, b

def predict(x, w, b):
    probs = softmax(x, w, b) # calculates probabilities of each class for the samples
    return np.argmax(probs, axis=1) # returns the indices of most probabilities which is the class

df = datasets.load_wine()  ## loading the wine dataset that has 3 classes
X = df.data
y = df.target

X, y = shuffle(X, y, random_state=1234) # shuffling the data to get all the classes in both train and test set

# using numpy to split the dataset
X_train, X_test = np.split(X, [int(0.8 * len(X))]) 
y_train, y_test = np.split(y, [int(0.8 * len(y))])

# scaling data for easy calculation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_samples, n_features = X_train.shape # X_train[n_samples, n_features]
n_classes = np.unique(y).size # getting the exact number of classes

y_encoded = onehot_encoded(y_train, n_samples, n_classes) # encoding to treat each class equally

w = np.zeros((n_features, n_classes)) 
# this is for the multiplication of w(features,classes) with each sample of x(1,features)

b = np.zeros((1, n_classes)) # this is for the sum with the product of x.w
# produces the result z(1, classes) that goes into the softmax function
# the function produces probabilties for each classes

lr = 0.001
epochs = 1000
u_w, u_b = gradient_descent(X_train, y_encoded, w, b, n_samples, lr, epochs) # storing updated weights and bias

pred = predict(X_test, u_w, u_b)
acc = np.mean(pred == y_test)
print('Accuracy: ', acc)

plt.scatter(X_test[:,0], pred, c=pred, cmap='prism')
plt.title('Multiclass Logistic Regression')
plt.show()