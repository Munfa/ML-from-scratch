import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.pyplot as plt

def sigmoid(x, w, b):
    ## z = w.x + b
    z = np.dot(x,w.T) + b
    sig = 1 / (1 + np.exp(-z)) # sigmoid function
    return sig

def compute_loss(y, pred):
    # loss = (1/n) * (-y*log(pred) - (1-y)*log(1-pred))
    loss = np.mean((-y * np.log(pred)) - ((1-y) * np.log(1-pred)))
    return loss

def gradient_descent(x, y, w, b, n_samples, lr, epochs):
    for epoch in range(epochs):
        pred = sigmoid(x, w, b) # predicting values to calculating loss
        loss = compute_loss(y, pred)
        dw = (1/n_samples)*(np.dot(x.T, (pred-y))) # derivative of the loss function with respect to w
        db = (1/n_samples)*(np.sum(pred - y))  # derivative of the loss function with respect to b
        w = w - (lr * dw)
        b = b - (lr * db)
        if (epoch % 100) == 0:
            print('Loss: ',loss) # printing loss after every 100 epoch

    return w, b

def predict(x, w, b):
    y_pred = sigmoid(x, w, b) # predicting values using the sigmoid function
    return y_pred

def visualization(x_test, y_test, w, b):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    x_1 = x_test[y_test == 1] # separating values with label 1
    x_0 = x_test[y_test == 0] # separating values with label 0

    pred_1 = predict(x_1, w, b) # predicting values for plotting
    pred_0 = predict(x_0, w, b)

    ### using the predicted values to view the curve
    ax.scatter(x_1[:, 0], pred_1, label='1')
    ax.scatter(x_0[:, 0], pred_0, label='0')
    ax.axhline(y=0.5, color='g', linestyle='--') # this is the threshold

    plt.title('Logistic Regression')
    plt.legend(loc='upper right')
    plt.show()

# using sklearn to make a toy dataset
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

X_train, X_test = np.split(X, [int(0.75 * len(X))]) # splitting data using numpy
y_train, y_test = np.split(y, [int(0.75 * len(y))])

n_samples, n_features = X_train.shape # samples = X_train[0] & features = X_train[1]
w = np.zeros(n_features)
b = 0
lr = 0.01
epochs = 500
updated_w, updated_b = gradient_descent(X_train, y_train, w, b, n_samples, lr, epochs)

pred = predict(X_test, updated_w, updated_b)
y_pred = [ 1 if y>0.5 else 0 for y in pred]  # classifying values based on threshold
accuracy = np.mean(y_pred == y_test)

print("Accuracy: ", accuracy)

visualization(X_test, y_test, updated_w, updated_b)