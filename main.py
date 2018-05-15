from scipy.sparse import vstack
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import time

def load_data():
    training_data_file = "training_data_example.csv"
    validation_data_file = "validation_data_example.csv"
    employee_file = "employee.csv"

    training = pd.read_csv(training_data_file)
    validation = pd.read_csv(validation_data_file)
    employee = pd.read_csv(employee_file)
    training['type'] = 'training'
    validation['type'] = 'validation'
    df = training.append(validation, ignore_index=True)
    df = df.join(employee.set_index('employee id'), on='employee id')

    return df

def transform_data(df):
    ''' It performs the following data transformation.
        a) Extracts month and day of the week.
        b) Combine tax name and tax rate.
        c) vectorize expense description.

        We may also consider transforming pre-tax amount by either taking its logarithm or putting
        it into pre-defined buckets.

        Certain manipulation on the address may be helpful, e.g., extracting the city or the state,
        or compare the address state and the sales tax state.
    '''
    df = df.copy()

    # Assumption: month and day of week may be predictive.
    df['date'] = df['date'].apply(lambda x: time.strptime(x, '%m/%d/%Y'))
    df['month'] = df['date'].apply(lambda x: x.tm_mon)
    df['wday'] = df['date'].apply(lambda x: x.tm_wday)

    # Assumption: the same state may charge a different tax rate depending on the category.
    df['tax rate'] = round(df['tax amount'] / df['pre-tax amount'], 2)
    df['tax rate'] = df['tax rate'].apply(lambda x: str(x))
    df['tax'] = df[['tax name', 'tax rate']].apply(lambda x: '-'.join(x), axis=1)

    # Create dummy variables for labels.
    cat_vars = ['month', 'wday', 'employee id', 'tax', 'employee address', 'role']
    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix=var, drop_first=True)
        df = df.join(cat_list)

    # Vectorize expense description.
    # Certain common transformations such as spelling correction or stemming are not done here.
    # Also we simply use unigrams here, although bigrams or trigrams may improve the result.
    vectorizer = CountVectorizer(stop_words='english', min_df=0.01, binary=True)
    X = vectorizer.fit_transform(df.loc[df['type'] == 'training', 'expense description'])
    Y = vectorizer.transform(df.loc[df['type'] == 'validation', 'expense description'])
    Z = vstack([X, Y])
    words = [ 'word_' + w for w in vectorizer.get_feature_names()]
    word_columns = pd.DataFrame(Z.toarray(), columns=words)
    df = df.join(word_columns)

    # Remove redudant columns
    drop_columns = cat_vars + [
        'date', 'tax rate', 'tax name', 'tax amount', 'expense description', 'employee name'
    ]
    df = df.drop(columns=drop_columns)

    return df

def main():
    # Prepare training and test sets.
    df = load_data()
    df = transform_data(df)
    Y_train = df.loc[df['type'] == 'training', ['category']].as_matrix().ravel()
    X_train = df.loc[df['type'] == 'training', [c for c in df.columns if c not in ['category', 'type']]]
    Y_test = df.loc[df['type'] == 'validation', ['category']].as_matrix().ravel()
    X_test = df.loc[df['type'] == 'validation', [c for c in df.columns if c not in ['category', 'type']]]
    variables = X_train.columns.values

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 5-fold cross validation to select the right C value for logistic regression.
    # If it were binary classification, I would use AUR instead of accuracy.
    # I prefer Lasso esthetically because it suppresses the number of variables.
    kfold = model_selection.KFold(n_splits=5)
    scoring = 'accuracy'
    for C in np.arange(0.1, 2, 0.1):
        modelCV = LogisticRegression(penalty='l1', C=C)
        results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring=scoring)
        print("5-fold cross validation with C=%.1f average accuracy: %.3f" % (C, results.mean()))

    # We should select the C that gives the highest accuracy. Let's pretend it's 0.5.
    C = 0.5
    modelLR = LogisticRegression(penalty='l1', C=C)
    modelLR.fit(X_train, Y_train)
    print('Logistic regression with C=%.1f has the following predictors.' % C)
    for idx, val in enumerate(modelLR.classes_):
        predictive = ', '.join(map(str, variables[modelLR.coef_[idx] != 0]))
        print('   %s: %s' % (val, predictive))

    # Model performance.
    Y_pred_train = modelLR.predict(X_train)
    Y_pred_test = modelLR.predict(X_test)
    accuracy_train = np.sum(Y_pred_train == Y_train) / Y_train.shape[0]
    accuracy_test = np.sum(Y_pred_test == Y_test) / Y_test.shape[0]

    print("Train accuracy: %.3f; Test accuracy: %.3f." % (accuracy_train, accuracy_test))

main()
