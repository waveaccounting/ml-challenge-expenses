import csv
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import numpy as np

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


seed = 123
file_emp = 'employee.csv'
file_train = 'training_data_example.csv'
file_valid = 'validation_data_example.csv'


def get_data():
    ip2employee = {}
    x_train, x_test = [], []
    labels_train, labels_test = [], []
    expense_desc_train, expense_desc_test = [], []

    with open(file_emp, 'rb') as fin_emp:
        reader = csv.reader(fin_emp)
        next(reader)
        for row in reader:
            id = row[0]
            address = row[2].lower().split(',')
            try:
                city = address[1].strip()
            except:
                city = ''
            try:
                province = address[2].split(' ')[1].strip()
            except:
                province = ''
            try:
                country = address[3].strip()
            except:
                country = ''

            role = row[3].lower()
            ip2employee[id] = {'city': city,
                            'province': province,
                            'country': country,
                            'role': role
                            }

    # note: reading training and validation files separately when their sizes are big
    # to avoid 'if' statement and reduce computing time
    for f in [file_train, file_valid]:
        with open(f, 'rb') as fin:
            reader = csv.reader(fin)
            next(reader)
            for row in reader:
                date = row[0]
                date_obj = datetime.strptime(date, '%m/%d/%Y')
                month = date_obj.month
                weekday = date_obj.weekday()

                category = row[1]

                id = row[2]
                description = row[3].lower()
                pre_tax = float(row[4])
                pre_tax_in_hundred = pre_tax / 100.0  # "normalize" to smaller values close to other input values
                tax_name = row[5].lower().split(' ')
                tax_area = tax_name[0]
                tax = float(row[6])
                tax_precentage = tax / pre_tax

                data = {
                    'month': month,
                    'weekday': weekday,
                    'pre_tax_in_hundred': pre_tax_in_hundred,
                    'tax_precentage': tax_precentage,
                    'tax_area': tax_area,
                    'role': ip2employee[id]['role'],
                    'city': ip2employee[id]['city'],
                    'province': ip2employee[id]['province'],
                    'country': ip2employee[id]['country']
                }

                if f == file_train:
                    labels_train.append(category)
                    x_train.append(data)
                    expense_desc_train.append(description)
                else:
                    labels_test.append(category)
                    x_test.append(data)
                    expense_desc_test.append(description)

    dv = DictVectorizer()
    x_train = dv.fit_transform(x_train)
    x_test = dv.transform(x_test)

    tv = TfidfVectorizer(stop_words='english')
    expense_desc_train = tv.fit_transform(expense_desc_train)
    expense_desc_test = tv.transform(expense_desc_test)

    x_train = sp.sparse.hstack([expense_desc_train, x_train])
    x_test = sp.sparse.hstack([expense_desc_test, x_test])

    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return ip2employee, x_train, x_test, y_train, y_test


def lr_tuning(dataset):
    (training_data, test_data) = dataset.randomSplit([0.8, 0.2], seed=seed)

    lr = LogisticRegression()
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction',
                                                  metricName='accuracy')

    param_grid = (ParamGridBuilder()
                  .addGrid(lr.regParam, [0.1, 0.3, 0.5])
                  .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.8])
                  .addGrid(lr.maxIter, [10, 20])
                  .build())

    # 5-fold cross validation
    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=5)

    cv_model = cv.fit(training_data)

    # predictions = cv_model.transform(test_data)
    # evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
    # result = evaluator.evaluate(predictions)
    # print result

    bestModel = cv_model.bestModel
    print '----------Parameter tuning results----------'
    print 'Best Param (regParam): ', bestModel._java_obj.getRegParam()
    print 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter()
    print 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam()
    print '--------------------------------------------'

    return bestModel


def category_prediction(x_train, x_test, y_train, y_test, mode=1):
    labels = list(set(y_train))
    class_num = len(labels)

    try:
        if mode == 1:
            # logistic regression using Spark
            sc = SparkContext('local', 'assignment')
            spark = SparkSession(sc)

            y_train = [labels.index(i) for i in y_train]
            y_test = [labels.index(i) for i in y_test]


            def create_df(x, y):
                dd = [(y[i], Vectors.dense(x.tocsr()[i].toarray().tolist())) for i in range(len(y))]
                df = spark.createDataFrame(sc.parallelize(dd), schema=['label', 'features'])

                return df

            df_train = create_df(x_train, y_train)
            df_test = create_df(x_test, y_test)

            # parameter tuning and cross validation
            model = lr_tuning(df_train)

            predictions = model.transform(df_test)
            pred = predictions.rdd.map(lambda r: r.prediction).collect()

            predictionAndLabels = sc.parallelize([(pred[i], float(y_test[i])) for i in range(len(pred))])

            metrics = MulticlassMetrics(predictionAndLabels)

            # overall stats
            precision = metrics.precision()
            recall = metrics.recall()
            f1Score = metrics.fMeasure()
            print '--------Logistic regression results---------'
            print('Precision = %s' % precision)
            print('Recall = %s' % recall)
            print('F1 Score = %s' % f1Score)

            # weighted stats
            print('Weighted recall = %s' % metrics.weightedRecall)
            print('Weighted precision = %s' % metrics.weightedPrecision)
            print('Weighted F(1) Score = %s' % metrics.weightedFMeasure())
            print('Weighted F(0.5) Score = %s' % metrics.weightedFMeasure(beta=0.5))
            print '--------------------------------------------'


        elif mode == 2:
            # neural network
            y_shape = class_num

            def vectorize_label(i):
                a = [0] * y_shape
                a[i] = 1.0
                return a

            y_train = [vectorize_label(labels.index(i)) for i in y_train]
            y_test = [vectorize_label(labels.index(i)) for i in y_test]

            x_train = x_train.todense()
            x_test = x_test.todense()

            x_shape = x_train.shape[1]

            model = Sequential()
            model.add(Dense(32, input_shape=(x_shape,), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(y_shape, activation='softmax'))

            # compile the model
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # fit the model
            batch_size = 8
            model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1)
            score_train = model.evaluate(x_train, y_train, verbose=0)
            print ('Training scores: loss %s, accuracy %s' % (score_train[0], score_train[1]))

            score_test = model.evaluate(x_test, y_test, verbose=0)
            print ('Validation scores: loss %s, accuracy %s' % (score_test[0], score_test[1]))


        else:
            print ('Please select a valid mode.')


    except ValueError:
        print ('Model error!')


def expense_type_prediction(ip2employee):
    # Some assumptions are:
    # The types (personal/business) of expenses are mainly based on the combination of
    # role, category, pre-tax amount and expense description.
    # eg. engineers do not have coffee expenses for business unless for client or team;
    # only IT and Admin employees have hardware/software expenses for business;
    # expenses with "client", "team" in the description are considered as business expenses, etc.
    # Preliminary labels are made based on the above assumptions.

    x_train, x_test = [], []
    labels_train, labels_test = [], []
    expense_desc_train, expense_desc_test = [], []

    for f in [file_train, file_valid]:
        with open(f, 'rb') as fin:
            reader = csv.reader(fin)
            next(reader)
            for row in reader:
                date = row[0]
                date_obj = datetime.strptime(date, '%m/%d/%Y')
                month = date_obj.month
                weekday = date_obj.weekday()
                category = row[1].lower()
                id = row[2]
                description = row[3].lower()
                pre_tax = float(row[4])
                pre_tax_in_hundred = pre_tax / 100.0
                tax_name = row[5].lower().split(' ')
                tax_area = tax_name[0]
                tax = float(row[6])
                tax_precentage = tax / pre_tax

                role = ip2employee[id]['role']
                data = {
                    'category': category,
                    'month': month,
                    'weekday': weekday,
                    'pre_tax_in_hundred': pre_tax_in_hundred,
                    'tax_precentage': tax_precentage,
                    'tax_area': tax_area,
                    'role': role
                }

                # expense type: 0-personal, 1-business
                expense_type = 1
                if 'client' in description or 'team' in description:
                    expense_type = 1
                elif role == 'engineer' and 'coffee' in description:
                    expense_type = 0
                elif role in ['sales', 'ceo', 'engineer'] and category in ['computer - software', 'computer - hardware']:
                    expense_type = 0

                if f == file_train:
                    labels_train.append(expense_type)
                    x_train.append(data)
                    expense_desc_train.append(description)
                else:
                    labels_test.append(expense_type)
                    x_test.append(data)
                    expense_desc_test.append(description)

    dv = DictVectorizer()
    x_train = dv.fit_transform(x_train)
    x_test = dv.transform(x_test)

    tv = TfidfVectorizer(stop_words='english')
    expense_desc_train = tv.fit_transform(expense_desc_train)
    expense_desc_test = tv.transform(expense_desc_test)

    x_train = sp.sparse.hstack([expense_desc_train, x_train]).todense()
    x_test = sp.sparse.hstack([expense_desc_test, x_test]).todense()

    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    x_shape = x_train.shape[1]

    model = Sequential()
    model.add(Dense(32, input_shape=(x_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # fit the model
    batch_size = 8
    model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1)
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print ('Training scores: loss %s, accuracy %s' % (score_train[0], score_train[1]))

    score_test = model.evaluate(x_test, y_test, verbose=0)
    print ('Validation scores: loss %s, accuracy %s' % (score_test[0], score_test[1]))



if __name__ == '__main__':
    ip2employee, x_train, x_test, y_train, y_test = get_data()

    # mode: 1-logistic regression on Spark, 2-neural network
    category_prediction(x_train, x_test, y_train, y_test, mode=2)

    expense_type_prediction(ip2employee)
