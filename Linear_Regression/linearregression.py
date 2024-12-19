import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

########## Calculating the Mean Squared Error ##########
def mean_squared_error(m, b, X_test, y_test):
    total_error = 0
    for i in range(len(X_test)):
        x = X_test[i] # X values
        y = y_test[i] # y values
        total_error += (y - (m*x + b)) ** 2  ### here (m*x+b) = y_predicted
    return total_error / float(len(X_test))

########## Finding m and b value to predict y_test value ##########
def gradient_descent(m_curr, b_curr, X_train, y_train, L=0.001):
    m_gradient = 0
    b_gradient = 0

    n = len(X_train)
    for i in range(n):
        x = X_train[i] # X value
        y = y_train[i] # y value
        y_pred = (m_curr * x) + b_curr
        diff = y - y_pred

        m_gradient += -(2/n) * x * diff 
        b_gradient += -(2/n) * diff

    m = m_curr - (m_gradient * L)
    b = b_curr - (b_gradient * L)
    
    return m, b 

########## Creating a dataset for Linear Regression ########### 
X, y = datasets.make_regression(n_samples=300, n_features=1, noise=10, random_state=3)

########## Splitting dataset into Train and Test sets using numpy ##########
X_train,X_test = np.split(X,[int(0.75 * len(X))])
y_train,y_test = np.split(y,[int(0.75 * len(y))])

########## Finding the best m and b value to minimize the error in predicting values for X_test ##########
m = 0 
b = 0
epochs = 300
for _ in range(epochs):
    m, b = gradient_descent(m, b, X_train, y_train, L=0.01)

########## Finding the error of our predictions ############
mse = mean_squared_error(m, b, X_test, y_test)
print('m = ', m)
print('b = ', b)
print('error = ', mse)

########## This line contains the predicted values for X_test ############
regression_line = [((m*x) + b) for x in X_test]
plt.scatter(X_test, y_test)
plt.plot(X_test, regression_line) 
plt.show()
