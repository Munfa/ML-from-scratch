import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SVM_SGD:
    def __init__(self, learning_rate=0.001, regular_param=0.1, n_iters=1000):
        self.lr = learning_rate
        self.C = regular_param
        self.n_iters = n_iters
        self.w= None
        self.b = None

    def hinge_loss(self, x, y):
        error = np.maximum(0, 1-(y*(np.dot(self.w, x)+self.b)))
        loss = (0.5 * np.linalg.norm(self.w)) + (self.C * np.sum(error))
        return loss

    def fit(self, X, y):
        n_features = X.shape[1]

        self.w = np.zeros(n_features)
        self.b = 0

        for iter in range(self.n_iters):
            for i, x_i in enumerate(X):
                condition = (y[i] * (np.dot(self.w, x_i) + self.b)) < 1
                if condition.any():
                    ##### misclassified
                    self.w -= self.lr * ((self.C * self.w) - (np.dot(y[i], x_i)))  ##### w -= lr * (C*w - yi*Xi)
                    self.b += self.lr * y[i]  ##### b += lr * yi

                else:
                    #### Correctly classified 
                    self.w -= self.lr * (self.C * self.w )  ##### w -= lr * (C*w)

                loss = self.hinge_loss(x_i, y[i])
            if (iter % 100) == 0:
                print("Epoch: ", iter, "\nLoss: ", loss)

    def predict(self, X):
        return np.sign(np.dot(self.w, X.T) + self.b)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def visulalization(self, x_train, y_train, x_test, pred):
        x_train_pos = x_train[y_train==1]
        x_train_neg = x_train[y_train==-1]

        x_test_pos = x_test[pred==1]
        x_test_neg = x_test[pred==-1]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        
        #plot positive training data
        self.ax.scatter(x_train_pos[:,0]+3, x_train_pos[:,1]+7, label='Positive Data', s=100)
        #plot negative training data
        self.ax.scatter(x_train_neg[:,0]+1, x_train_neg[:,1]-9, label='Negative Data', s=100)

        # #plot positive predicted data
        self.ax.scatter(x_test_pos[:,0]+3, x_test_pos[:,1]+4, label='Positive Predicted Data', s=100, marker='*')
        #plot negative predicted data
        self.ax.scatter(x_test_neg[:,0]+1, x_test_neg[:,1]-4, label='Negative Predicted Data', s=100, marker='*')

        def hyperplane(x,w,b,v):
            return ((-w[0]*x)-b+v) / w[1]
        
        x_train_max = np.max(x_train)
        x_train_min = np.min(x_train)
        datarange = (x_train_min*0.3, x_train_max*0.5)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive support vector margin
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max], [psv1,psv2], 'b')

        # negative support vector margin
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2], 'b')

        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max], [db1,db2], 'k--')

        plt.title("SVM with SGD")
        plt.legend(loc='upper right')
        plt.show()

    
if __name__ == "__main__":

    df = datasets.load_breast_cancer()
    X = df.data
    y = df.target
    
    ## converting y={0,1} to y={-1,1}
    y = np.where(y==0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVM_SGD()
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    print("Accuracy:", svm.accuracy(X_test, y_test))
    
    svm.visulalization(X_train, y_train, X_test, y_pred) 



    