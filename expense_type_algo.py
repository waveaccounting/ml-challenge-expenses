import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Read from CSV
print 'Reading from CSV files....'
df = pd.read_csv('training_data_example_expense_type.csv')
df.head()

# Create dataframe to be used to train model to identify keywords in description
col = ['expense_type', 'expense_description']
df = df[col]
df = df[pd.notnull(df['expense_description'])]
df.columns = ['expense_type', 'expense_description']

# Transform the symbolic values into for classes to values suitable for the classifier
df['category_id'] = df['expense_type'].factorize()[0]
category_id_df = df[['expense_type', 'category_id']].drop_duplicates().sort_values('category_id')

# Prepare dictionaries to be used to identify key words
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'expense_type']].values)
df.head()

# fig = plt.figure(figsize=(8,6))
# df.groupby('expense_type').expense_description.count().plot.bar(ylim=0)
# plt.show()

# Implement bag of words model and TF-IDF measure (Term Frequency-Inverse Document Frequency)
# Use Stop Words to exclude irrelevant 'noise' data (e.g. pronouns)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                        ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.expense_description).toarray()
labels = df.category_id
features.shape

# Code below can be used to find the words that are the most correlated
# with the expense type:
# N = 2
# for expense_type in sorted(category_to_id.items()):
#   features_chi2 = chi2(features, labels == df.category_id)
#   indices = np.argsort(features_chi2[0])
#   feature_names = np.array(tfidf.get_feature_names())[indices]
#   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#   print("# '{}':".format(expense_type))
#   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# Make Naive Bayes classification model
X_train, X_test, y_train, y_test = train_test_split(df['expense_description'], df['expense_type'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Put in possible description text to see algorithm & classifier in action :)
term_to_test = "business dinner"
print(clf.predict(count_vect.transform([term_to_test])))