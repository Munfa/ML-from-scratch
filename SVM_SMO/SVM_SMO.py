import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SVM_SMO:
    def __init__(self, regular_param=1.0, tol=1e-3, n_iters=100):
        self.C = regular_param ## regularizing parameter
        self.tol = tol ## tolerance  
        self.n_iters = n_iters ## iterations

    def kernel(self, x1, x2):
        return np.dot(x1,x2) #Linear kernel

    ## calculating error
    def error(self, x, y):
        return self.predict(x) - y

    ## calculating L & H 
    def calc_L_H(self, i, j, y1, y2):
        if y1 != y2:
            ## L = max(0, alpha2 - alpha1)
            ## H = min(C, C + alpha2 - alpha1)
            return (max(0, self.alpha[j] - self.alpha[i]), min(self.C, self.C + self.alpha[j] - self.alpha[i]))

        else:
            ## L = max(0, alpha2 + alpha1 - C)
            ## H = min(C, alpha2 + alpha1)
            return (max(0, self.alpha[i] + self.alpha[j] - self.C), min(self.C, self.alpha[j] + self.alpha[i]))

    ## calculating weights and bias after getting optimal value of alpha
    def calc_w_b(self, x, y):
        ## w = alpha_i * yi * xi
        self.w = np.dot((self.alpha * y), x) 
        support_vectors = self.alpha > 0 ## 0 < alpha < C - KKT conditions
        ## b = (ys - xs * w) / n
        self.b = np.mean(y[support_vectors] - np.dot(x[support_vectors], self.w)) 

    def fit(self, x, y):
        n_samples,n_features = x.shape
        self.alpha = np.zeros(n_samples)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            alpha_prev = np.copy(self.alpha)
            
            ## choosing different i and j for alpha_i and alpha_j
            for i in range(n_samples):
                j = np.random.randint(0, n_samples)
                if i == j:
                    continue

                x_i, x_j = x[i, :], x[j, :]
                y_i, y_j = y[i], y[j]

                ## eta = K(x1, x1) + K(x2, x2) - 2K(x1, x2)
                eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - (2 * self.kernel(x_i, x_j)) ## needed for updating alpha
                if eta == 0:
                    continue
                E_i = self.error(x_i, y_i) # calculating errors
                E_j = self.error(x_j, y_j) # to calculate new alpha_j

                # alpha2_new = alpha2 + (y2 * (E1 - E2)) / eta
                alpha_j_new = self.alpha[j] + (y_j * (E_i - E_j)) / eta 

                L, H = self.calc_L_H(i, j, y_i, y_j) # calculating L & H for clipping alpha2_new

                ##clipping alpha_j to update alpha_i
                alpha_j_new = np.clip(alpha_j_new, L, H) ## alpha2_new_clipped
                if abs(alpha_j_new - self.alpha[j]) < self.tol: ## tol checks alpha values to help reach convergence faster
                    continue
                self.alpha[j] = alpha_j_new
                ## alpha1_new = alpha1 + (y1*y2)(alpha2 - alpha2_new_clipped)
                self.alpha[i] += y_i * y_j * (self.alpha[j] - alpha_j_new) 

            if np.allclose(self.alpha, alpha_prev, atol=self.tol): ## checks if both alphas are equal
                break
            self.calc_w_b(x, y)

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)
    
    def visualization(self, x, y, test, pred):
        x_pos = x[y==1]     # separating positive
        x_neg = x[y==-1]    # and negative data points based on the y labels

        test_pos = test[pred==1] # separating data based on predicted labels
        test_neg = test[pred==-1]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        ### Positive and negative training data
        self.ax.scatter(x_pos[:, 0]+1, x_pos[:, 1]-4, s=100, label='Positive Data')
        self.ax.scatter(x_neg[:, 0]+1, x_neg[:, 1]+2, s=100, label='Negative Data')

        ### Positive and negatice predicted data
        self.ax.scatter(test_pos[:, 0]+1, test_pos[:, 1]-4, s=100, label='Positive Predicted Data', marker='*')
        self.ax.scatter(test_neg[:, 0]+1, test_neg[:, 1]+2, s=100, label='Negative Predicted Data', marker= '*')

        ## w.x + b = 0
        ## w1x1 + w2x2 + b = 0
        ## x2 = (-w1x1 - b) / w2
        def hyperplane(x, w, b, v):
            return ((-w[0] * x)-b+v) / w[1]
        
        x_min = np.min(x)
        x_max = np.max(x)
        datarange= (x_min*0.3, x_max*0.4) ## picking datapoint for the graph
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive support vector margin
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1-3, psv2-3], 'b--')

        # negative support vector margin
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1+3, nsv2+3], 'r--')

        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'k')

        plt.title('SVM with SMO')
        plt.legend(loc='upper right')
        plt.show()


df = datasets.load_breast_cancer()
X = df.data
y = df. target

y = np.where(y==0, -1, 1)

# splitting dataset using numpy
X_train, X_test = np.split(X, [int(0.8 * len(X))])
y_train, y_test = np.split(y, [int(0.8 * len(y))])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVM_SMO()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy: ", accuracy)

svm.visualization(X_train, y_train, X_test, y_pred)