from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def data_matrix(t_data, v_data, e_data):
    """
    Converts Training (t_data) and Validating (v_data) 
    data into numpy arrays.
    """
    le = preprocessing.LabelEncoder()
    # employee role
    le.fit(e_data['role']) 
    e_role = le.transform(e_data.loc[t_data['employee id']]['role'])
    val_e_role = le.transform(e_data.loc[v_data['employee id']]['role'])
    
    le.fit(e_data.index.values) # employee id is the index column 
    e_id = le.transform(t_data['employee id'])
    val_e_id = le.transform(v_data['employee id'])

    # Type of Tax encoded
    le.fit(t_data['tax name'])
    tax_name = le.transform(t_data['tax name'])
    val_tax_name = le.transform(v_data['tax name'])

    Xtrain = np.c_[e_role, e_id, t_data['pre-tax amount'], t_data['tax amount'], tax_name]
    Xval = np.c_[val_e_role, val_e_id, v_data['pre-tax amount'], v_data['tax amount'], val_tax_name]
    return (Xtrain, Xval)
