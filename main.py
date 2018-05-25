import csv
import os

import numpy as np
import pandas as pd
import self as self
from sklearn.model_selection import KFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

pd.options.mode.chained_assignment = None
bus = pd.DataFrame()
print 'Reading from CSV files....'
train = pd.read_csv("training_data_example.csv")
valid = pd.read_csv("validation_data_example.csv")
test = pd.read_csv("training_data_example_and_validation.csv")

# Transform the symbolic values into numbers suitable for the classifier
print 'Doing some data pre-processing/cleaning....'
train['category'] = pd.factorize(train['category'])[0]
valid['category'] = pd.factorize(valid['category'])[0]
test['category'] = pd.factorize(test['category'])[0]
train['expense description'] = pd.factorize(train['expense description'])[0]
valid['expense description'] = pd.factorize(valid['expense description'])[0]
test['expense description'] = pd.factorize(test['expense description'])[0]
train['tax name'] = pd.factorize(train['tax name'])[0]
valid['tax name'] = pd.factorize(valid['tax name'])[0]
test['tax name'] = pd.factorize(test['tax name'])[0]
train['date'] = pd.factorize(train['date'])[0]
valid['date'] = pd.factorize(valid['date'])[0]
test['date'] = pd.factorize(test['date'])[0]
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Format the data and expected values for SKLearn
trainData = pd.DataFrame(train[['expense description', 'tax amount', 'date']])
trainTarget = np.array(pd.DataFrame(train[['category']]))
testData = pd.DataFrame(test[['expense description', 'tax amount', 'date']])
testTarget = pd.DataFrame(test[['category']])
valData = pd.DataFrame(valid[['expense description', 'tax amount', 'date']])
valTarget = pd.DataFrame(valid[['category']])

# Prepare cross-validation folds & variables
k_fold = KFold(len(valData), shuffle=True, random_state=0)
algoEval = 0
winningAlgo = ""

# Change y vectors to 1d array
trainTarget = np.ravel(trainTarget, 'C')
valTarget = np.ravel(valTarget, 'C')

# Start cross-validation of candidate algorithms
print 'Evaluating model algorithms for dataset....'
GBCclassifier = GradientBoostingClassifier()
GBCclassifier.fit(trainData, trainTarget)
if algoEval < cross_val_score(GBCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
    algoEval = cross_val_score(GBCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
    classifier = GradientBoostingClassifier()
    winningAlgo = 'Gradient Boost'
print 'GBC:  ', cross_val_score(GBCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
GNBclassifier = GaussianNB()
GNBclassifier.fit(trainData, trainTarget)
if algoEval < cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
    algoEval = cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
    classifier = GNBclassifier()
    winningAlgo = 'Gaussian Naive Bayes'
print 'GNB:  ', cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
SVCclassifier = SVC()
SVCclassifier.fit(trainData, trainTarget)
if algoEval < cross_val_score(SVCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
    algoEval = cross_val_score(SVCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
    classifier = SVCclassifier()
    winningAlgo = 'Support Vector Machine'
print 'SVM:  ', cross_val_score(SVCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
LDAclassifier = LinearDiscriminantAnalysis()
LDAclassifier.fit(trainData, trainTarget)
if algoEval < cross_val_score(LDAclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
    algoEval = cross_val_score(LDAclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
    classifier = LDAclassifier()
    winningAlgo = 'Linear Discriminant Analysis'
print 'LDA:  ', cross_val_score(LDAclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
LinREGclassifier = LinearRegression()
LinREGclassifier.fit(trainData, trainTarget)
if algoEval < cross_val_score(LinREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
    algoEval = cross_val_score(LinREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
    classifier = LinREGclassifier()
    winningAlgo = 'Linear Regression'
print 'LinReg:  ', cross_val_score(LinREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
LogREGclassifier = LogisticRegression()
LogREGclassifier.fit(trainData, trainTarget)
if algoEval < cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
    algoEval = cross_val_score(LogREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
    classifier = LogREGclassifier()
    winningAlgo = 'Logistic Regression'
print 'LogReg:  ', cross_val_score(LogREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
print '\n the best algorithm is ' + winningAlgo + ', proceeding to model test: \n'

classifier.fit(trainData, trainTarget)
predictedValues = classifier.predict(testData)
# print("\n\nTest set expenses:")
# print(itemfreq(predictedValues))

testResults = test[['employee id']]
testResults['category'] = predictedValues
# print(testResults.head())

print 'Model predicted with ', accuracy_score(testTarget, predictedValues), \
    ' accuracy, check prediction.csv & detailed_result_output.csv for details.'
detailed_result = classification_report(testTarget.values.flatten(), predictedValues,
                                        target_names=['Travel', 'Meals and Entertainment',
                                                      'Computer - Hardware', 'Computer - Software', 'Office Supplies'])

# Save results to files
with open(os.path.join('detailed_result.csv'), 'w') as tr:
    tr.write(detailed_result)
testResults.to_csv('prediction.csv', index=False)
