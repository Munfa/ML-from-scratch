import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm

# Declaring empty dictionaries to store mean. variance and prior probabilities of the classes
means = {}
vars = {}
priors = {}

def calc_mean_var_prior(x, y):
    classes = np.unique(y)                  # returns the classes/labels
    for c in classes:
        rows = x[y==c]                      # rows of class c
        means[c] = np.mean(rows, axis=0)    # average of the feature values that belongs to class c
        vars[c] = np.var(rows, axis=0)      # variance shows how far the values are from the mean
        priors[c] = rows.shape[0] / x.shape[0] # probability of a class out of all the classes
    
    return means, vars, priors

def gaussian_prob(x, mean, var):
    '''
        Gaussian formula is used to calculate the likelihood of a value being in the class c
        Based on the mean and variance we know how likely that value is in the class c
    '''
    eps = 1e-9 # a very small number to avoid division by zero
    exponent = np.exp(-(x-mean)**2 / (2* var + eps))    # calculating the eponential separately
    likelihood = (1/ np.sqrt(2*np.pi*var + eps)) * exponent
    return likelihood

def predict(Xs, y):
    classes = np.unique(y)  # returns the labels/classes
    preds = []              # an empty list to store the predictions
    for x in Xs:
        probs = {}          # an empty dictionary to store probabilities of each class
        for c in classes:
            ''' 
                we take the log of the result to simplify the math and avoid multiplying with zero
                as the probabilities may return a very small number which can be accepted as a zero
            '''
            prior = np.log(priors[c])   # getting the calculated prior probability of a class from priors dict
            likelihood = np.sum(np.log(gaussian_prob(x, means[c], vars[c])))    # calculating likelihood
            probs[c] = prior + likelihood   # as we use log the multiplication becomes addition
        pred_class = max(probs, key=probs.get)  # the class that has the maximum probability is the prediction
        preds.append(pred_class)    # append the predictions of each feature
    return preds

df = datasets.load_iris()   # using the Iris dataset
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) # splitting data into train and test set

means, vars, priors = calc_mean_var_prior(X_train, y_train) # calculating mean, variance and prior for the training data
pred = predict(X_test, y_test)

accuracy = np.mean(pred == y_test)
print('Accuracy= ',accuracy)

fig = plt.figure()
ax = fig.add_subplot()
# ax.scatter(range(len(y_test)), y_test, marker='o', label='actual')
# ax.scatter(range(len(pred)), pred, marker='x', label='predicted')
# plt.title('Naive Bayes')

####### plot distribution ######
for cls in means:
    stds = {}
    stds[cls] = np.sqrt(vars[cls])  # calculating standard deviation for plot
    mean = means[cls][0]
    std = stds[cls][0]
    x = np.linspace(mean - 4*std, mean + 4*std, 200)    # using linspace for a smooth curve; it generates a sequence of numbers
    y = norm(loc=mean, scale=std).pdf(x)    # pdf = Probability Density Function
    ax.plot(x, y, label=f'Class {cls}')
plt.title('Gaussian Distributions')
plt.xlabel('Feature Values')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')
plt.show()
