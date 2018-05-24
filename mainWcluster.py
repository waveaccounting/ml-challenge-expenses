from ipyparallel import Client

# Set up cluster clients to simulate distributed computing
# NOTE: Make sure that a local cluster is configured and running,
# and also make sure the source csv files are in the default local
# cluster engine path at /Python27/Lib/site-packages/ipyparallel
clients = Client()
rc = clients.load_balanced_view()

# function to be called to implement on cluster client(s),
# containing the actual ML code to run
def clustering():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    pd.options.mode.chained_assignment = None

    print 'Reading from CSV files....'
    train = pd.read_csv("training_data_example.csv")
    valid = pd.read_csv("validation_data_example.csv")
    test = pd.read_csv("training_data_example_and_validation.csv")

    # Transform the symbolic values into numbers suitable for the Bayes classifier
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
    train.fillna(train.mean(), inplace=True)
    test.fillna(test.mean(), inplace=True)

    # Format the data and expected values for SKLearn
    trainData = pd.DataFrame(train[['expense description', 'pre-tax amount', 'tax name', 'tax amount']])
    trainTarget = np.array(pd.DataFrame(train[['category']]))
    testData = pd.DataFrame(test[['expense description', 'pre-tax amount', 'tax name', 'tax amount']])
    testTarget = pd.DataFrame(test[['category']])
    valData = pd.DataFrame(valid[['expense description', 'pre-tax amount', 'tax name', 'tax amount']])
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
    print 'GBC:  ', cross_val_score(GBCclassifier, valData,valTarget, cv=k_fold, n_jobs=1).mean() * 100
    GNBclassifier = GaussianNB()
    GNBclassifier.fit(trainData, trainTarget)
    if algoEval < cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
        algoEval = cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
        classifier = GNBclassifier()
        winningAlgo = 'Gaussian Naive Bayes'
    print 'GNB:  ', cross_val_score(GNBclassifier, valData,valTarget, cv=k_fold, n_jobs=1).mean() * 100
    SVCclassifier = SVC()
    SVCclassifier.fit(trainData, trainTarget)
    if algoEval < cross_val_score(SVCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
        algoEval = cross_val_score(SVCclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
        classifier = SVCclassifier()
        winningAlgo = 'Support Vector Machine'
    print 'SVM:  ', cross_val_score(SVCclassifier, valData,valTarget, cv=k_fold, n_jobs=1).mean() * 100
    LDAclassifier = LinearDiscriminantAnalysis()
    LDAclassifier.fit(trainData, trainTarget)
    if algoEval < cross_val_score(LDAclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
        algoEval = cross_val_score(LDAclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
        classifier = LDAclassifier()
        winningAlgo = 'Linear Discriminant Analysis'
    print 'LDA:  ', cross_val_score(LDAclassifier, valData,valTarget, cv=k_fold, n_jobs=1).mean() * 100
    LinREGclassifier = LinearRegression()
    LinREGclassifier.fit(trainData, trainTarget)
    if algoEval < cross_val_score(LinREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
        algoEval = cross_val_score(LinREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
        classifier = LinREGclassifier()
        winningAlgo = 'Linear Regression'
    print 'LinReg:  ', cross_val_score(LinREGclassifier, valData,valTarget, cv=k_fold, n_jobs=1).mean() * 100
    LogREGclassifier = LogisticRegression()
    LogREGclassifier.fit(trainData, trainTarget)
    if algoEval < cross_val_score(GNBclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100:
        algoEval = cross_val_score(LogREGclassifier, valData, valTarget, cv=k_fold, n_jobs=1).mean() * 100
        classifier = LogREGclassifier()
        winningAlgo = 'Logistic Regression'
    print 'LogReg:  ', cross_val_score(LogREGclassifier, valData,valTarget, cv=k_fold, n_jobs=1).mean() * 100
    print '\n the best algorithm is '+winningAlgo+', proceeding to model test: \n'

    classifier.fit(trainData, trainTarget)
    predictedValues = classifier.predict(testData)

    testResults = test[['employee id']]
    testResults['category'] = predictedValues

    print 'Model predicted with ', accuracy_score(testTarget, predictedValues),\
        ' accuracy, check prediction.csv for details.'

    # As this is a cluster simulation, the file will be saved on
    # the default path for local engine at /Python27/Lib/site-packages/ipyparallel
    testResults.to_csv('prediction.csv', index=False)

# Calling the function code onto the cluster
rc.block = True
rc.apply(clustering)
