from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, classification_report


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
        y_pred = self.model.predict(self.data)
        return y_pred

    @property
    def __str__(self):
        accuracy = accuracy_score(self.label, self.predict)
        return "Accuracy = {0:.2f} \n{1}".format(accuracy, classification_report(self.label, self.predict))


class DimensionalityReduction:
    def __init__(self, data):
        self.data = data

    @property
    def pca(self):
        model = PCA(n_components=2)
        model.fit(self.data)
        return model

    @property
    def tsne(self):
        model = TSNE(n_components=2)
        model.fit(self.data)
        return model


class Clustering:
    def __init__(self, data):
        self.data = data

    @property
    def kmeans(self):
        model = KMeans(n_clusters=2)
        model.fit(self.data)
        return model
