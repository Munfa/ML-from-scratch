import numpy as np
from sklearn import datasets

def calc_mean_var_prior(x, y):
    means = {}
    vars = {}
    priors = {}
    classes = np.unique(y)
    for c in classes:
        rows = x[y==c]                      # rows of class c
        means[c] = np.mean(rows, axis=0)    # 
        vars[c] = np.std(rows, axis=0)
        priors[c] = rows.shape[0] / x.shape[0]
    
    return means, vars, priors

def gaussian_prob(x, mean, var):
    eps = 1e-9 # a very small number to avoid division by zero
    exponent = np.exp(-(x-mean)**2 / (2* var + eps))
    prob = (1/ np.sqrt(2*np.pi*var + eps)) * exponent
    return prob

df = datasets.load_iris()
X = df.data
y = df.target
means, vars, priors = calc_mean_var_prior(X, y)
