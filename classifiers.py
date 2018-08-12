from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class Train:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    @property
    def random_forest(self):
        model = RandomForestClassifier(n_estimators=5, criterion='entropy')
        model.fit(self.data, self.label)
        return model

    @property
    def svm(self):
        model = SVC(C=2, degree=2, gamma=0.1, kernel='rbf', probability=True)
        model.fit(self.data, self.label)
        return model

    @property
    def knn(self):
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        model.fit(self.data, self.label)
        return model

    @property
    def logistic_regression(self):
        model = linear_model.LogisticRegression(C=10.0)
        model.fit(self.data, self.label)
        return model


class Validate:
    def __init__(self, model, data, label):
        self.model = model
        self.data = data
        self.label = label

    @property
    def predict(self):
        y_pred_train = self.model.predict(self.data)
        accuracy = accuracy_score(self.label, y_pred_train)
        return accuracy
