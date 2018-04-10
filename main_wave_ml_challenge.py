
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import sklearn 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[29]:


# read training, validation and employees data files into pandas dataframe
training_data = pd.read_csv("./training_data_example.csv")
validation_data = pd.read_csv("./validation_data_example.csv")
employee_data = pd.read_csv("./employee.csv")

training_data = pd.merge(training_data, employee_data, how="inner",on="employee id").drop(['employee address','employee name'],axis=1)

validation_data = pd.merge(validation_data, employee_data, how="inner",on="employee id").drop(['employee address','employee name'],axis=1)

# concatenate training and validation data for feature engineering in order to have consistency of columns
training_data['type'] = "training"
validation_data['type'] = "validation"
combined_data = pd.concat([training_data, validation_data], ignore_index = True)


# In[30]:


# summary of training data 
combined_data['employee id'] = combined_data['employee id'].astype(str)
# drop date column, since we are not considering it as a forecasting problem
combined_data = combined_data.drop(['date'],axis=1)
print(combined_data.describe(include="all"))


# In[32]:


# normalize real value features 
min_max_scaler = MinMaxScaler()

combined_data['pre-tax amount'] = min_max_scaler.                            fit_transform(np.array(combined_data['pre-tax amount']).reshape(-1, 1))
combined_data['tax amount'] = min_max_scaler.                            fit_transform(np.array(combined_data['tax amount']).reshape(-1, 1))       


# In[33]:


# get one hot encoded training and validation data for categorical variables 'tax name' and 'role'
combined_data = pd.get_dummies(combined_data, columns=['tax name', 'role'], drop_first=True)
combined_data.columns


# In[34]:


# for 'expense description' column generate vocabulary to better parse the description for classification task
#pd.get_dummies(training_data, sep=' ', columns=['expense description'], drop_first=True)
# first remove stop words to reduce the dimensionality of features thus generated
# and convert all words to lower case to disambiguate 
stop = stopwords.words('english')
combined_data['expense_desc_stop_removed'] = combined_data['expense description'].apply(lambda x: ' '.                                            join([word.lower() for word in x.split() if word not in (stop)]))


# then generate vocabulary for all words that occur under expense description column and assign 

print("vocabulary generated from expense description column")
print(combined_data['expense_desc_stop_removed'].str.get_dummies(sep=' ').columns)
df_vocab_expenses_ =  combined_data['expense_desc_stop_removed'].str.get_dummies(sep=' ')

joined_train_val_ = combined_data.merge(df_vocab_expenses_, how='outer', left_index=True, right_index=True)
X_train = joined_train_val_[joined_train_val_.type=="training"]
X_validation = joined_train_val_[joined_train_val_.type=="validation"]
X_train.columns


# In[35]:


# get target/label data by encoding 'category' column
Y_train = X_train['category'] 
le_targets = LabelEncoder()
le_targets.fit(Y_train)
Y_train = le_targets.transform(Y_train)

Y_validation = X_validation['category']
le_validation_targets = LabelEncoder()
le_validation_targets.fit(Y_validation)
Y_validation = le_validation_targets.transform(Y_validation)


# In[36]:


# filter out the columns not to be used as training feature set
drop_cols = ['category', 'employee id', 'expense description', 'expense_desc_stop_removed','type']
X_train = X_train.drop(drop_cols, axis=1)
X_validation = X_validation.drop(drop_cols, axis=1)
X_validation.columns


# In[37]:


# build classifier
rand_forest = RandomForestClassifier()
rand_forest.fit(X_train.values, Y_train)
rand_forest.feature_importances_


# In[38]:


# cross validation to avoid overfitting as its often the case from small datasets
scores = ['precision', 'recall']
# Set the parameters by cross-validation
tuning_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [1, 10, 100]},
                    {'kernel': ['linear'], 'C': [1, 10, 100], 'gamma': [1e-2, 1e-3, 1e-4]}]
scores = ['precision', 'recall', 'f1']
           


# In[39]:


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuning_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, Y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_validation, clf.predict(X_validation)
    print(classification_report(y_true, y_pred))
    print()
    


# In[40]:


best_estimator = clf.best_estimator_
Y_predicted_on_validation = best_estimator.predict(X_validation)


# In[353]:





# In[352]:


training_data


# In[1]:


X_train_clustering = training_data[['pre-tax amount','tax amount','category']]
X_train_clustering['category'] = le_targets.transform(Y_train)


# In[58]:


# normalize real value features 
min_max_scaler = MinMaxScaler()
training_data_clustering = training_data
training_data_clustering['pre-tax amount'] = min_max_scaler.                            fit_transform(np.array(training_data['pre-tax amount']).reshape(-1, 1))
training_data_clustering['tax amount'] = min_max_scaler.                            fit_transform(np.array(training_data['tax amount']).reshape(-1, 1)) 
training_data_clustering = pd.get_dummies(training_data_clustering, columns=['tax name', 'role','category'],                                           drop_first=True)
# drop columns 
training_data_clustering = training_data_clustering.drop(['employee id','date','type','expense description'],axis=1).                                    astype('float')
X_train_clustering = training_data_clustering.values

X_train_clustering.shape


# In[61]:


# Clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train_clustering)
kmeans.labels_


# In[64]:


validation_data


# In[ ]:


db = DBSCAN(eps=0.8, min_samples=2).fit(training_data[['pre-tax amount','tax amount']].values)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

