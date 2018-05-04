import csv
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

class Classifier:
    
    def __init__(self, name = '', **kwargs):
        if name == 'random':
            self.model = RandomForestClassifier(**kwargs)
        elif name == 'trees':
            self.model = ExtraTreesClassifier(**kwargs)
        elif name == 'logistic':
            self.model = LogisticRegression(**kwargs)
        elif name == 'lasso':
            self.model = Lasso(**kwargs)
        elif name == 'l2reg':
            self.model = Ridge(**kwargs)
        elif name == 'svm':
            self.model = SVC(verbose = True, **kwargs)
        elif name == 'xgboost':
            self.model = XGBClassifier(silent = True, **kwargs)
        elif name == 'neural_network':
            self.model = MLPClassifier(random_state = 42, **kwargs)
        elif name == 'knn':
            self.model = KNeighborsClassifier(**kwargs)
        else:
            self.model = LogisticRegression(n_jobs = 5)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def grid_search(self, X, Y, params, folds = 20):
        self.model = GridSearchCV(self.model, params, n_jobs = 2, verbose = 2, cv = folds)
        self.model.fit(X, Y)
        print "Best Estimator: ", self.model.best_estimator_
        print "Best Parameter: ", self.model.best_params_
        print "Best Score: ", self.model.best_score_
        np.save('../data/cv_results.npy', self.model.cv_results_)

    def predict(self, X):
        return self.model.predict(X)

    def rpredict(self, Xtrain, Ytrain, Xtest):
        ## TODO: Get the model parameters accepted and retrain
        self.model.fit(Xtrain, Ytrain, eval_metric = r2_score)
        return self.model.predict(Xtest)

    def score(self, ytrue, ypred):
        r = r2_score(ytrue, ypred)
        print "R2 Score: ", r
        return r

    def write_csv(self, ypred, Ids = None, columns = ['ID', 'y']):
        ofile = open(PREDICTION_FILE, "wb") 
        writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        row = []
        for c in columns:
            row.append(c)
        writer.writerow(row)

        if not Ids:
            Ids = range(len(ypred))

        for count, y in zip(Ids, ypred):
            row = []
            row.append(count)
            row.append(y)
            writer.writerow(row)

        ofile.close()
