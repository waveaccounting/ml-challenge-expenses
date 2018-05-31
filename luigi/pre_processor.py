from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import luigi
from luigi import LocalTarget
import pandas as pd
import numpy as np
import pickle
import os


class PreProcess(luigi.Task):
    """
    This class will pre-process and vectorize the data
    and output to CSVs and pickles
    """

    output_path = luigi.Parameter()
    input_path = luigi.Parameter()

    def output(self):
        """        
        :return: a dictionary of 5 files:
        vectorized training data
        vectorized validation data
        vectorized training data with only tf-idf features
        vectorized validation data with only tf-idf features
        a labels files including training and validation labels
        """
        return {
            'train_x': LocalTarget(os.path.join(self.output_path, 'train_processed.csv')),
            'train_x_tfidf': LocalTarget(os.path.join(self.output_path, 'train_processed_tfidf.csv')),
            'validation_x': LocalTarget(os.path.join(self.output_path, 'validation_processed.csv')),
            'validation_x_tfidf': LocalTarget(os.path.join(self.output_path, 'validation_processed_tfidf.csv')),
            'labels': LocalTarget(os.path.join(self.output_path, 'labels.pkl'))
        }

    def pre_process(self, df, df_employee,
                    columns_to_drop=['date',
                                     'category',
                                     'tax amount',
                                     'expense description']):

        df['day_of_week'] = pd.to_datetime(df['date']).apply(lambda x: x.weekday()).astype(
            str)  # str so that treated as categoical
        df['month'] = pd.to_datetime(df['date']).apply(lambda x: x.month).astype(str)
        df = pd.merge(df, df_employee[['employee id', 'role']], how='inner', on=['employee id'])
        df['employee id'] = df['employee id'].astype(str)
        df = df.drop(columns_to_drop, axis=1)

        # one-hot encode the categorical variables
        df = pd.get_dummies(df)

        df['pre-tax amount'] = preprocessing.minmax_scale(df[['pre-tax amount']])

        return df

    def run(self):
        df_train = pd.read_csv(os.path.join(self.input_path, 'training_data_example.csv'))
        df_val = pd.read_csv(os.path.join(self.input_path, 'validation_data_example.csv'))
        df_employee = pd.read_csv(os.path.join(self.input_path, 'employee.csv'))

        x_train = self.pre_process(df_train, df_employee)
        x_val = self.pre_process(df_train, df_employee)
        x_train, x_val = x_train.align(x_val, join='left', axis=1)
        x_val = x_val.fillna(0)

        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit(df_train['expense description'])
        x_train_tfidf = vectorizer.transform(df_train['expense description']).toarray()
        x_val_tfidf = vectorizer.transform(df_val['expense description']).toarray()

        lencoder = LabelEncoder()
        lencoder.fit(df_train['category'])
        names = set(df_val['category'])  # label names to be used later
        y_train = lencoder.transform(df_train['category'])
        y_val = lencoder.transform(df_val['category'])

        val_categories = []
        for clazz in lencoder.classes_:
            if clazz in names:
                val_categories.append(clazz)

        x_train.to_csv(self.output()['train_x'].path, sep=',')
        x_val.to_csv(self.output()['validation_x'].path, sep=',')
        np.savetxt(self.output()['train_x_tfidf'].path, x_train_tfidf, delimiter=",")
        np.savetxt(self.output()['validation_x_tfidf'].path, x_val_tfidf, delimiter=",")

        labels = {
            'y_train': y_train,
            'y_val': y_val,
            'categories': lencoder.classes_,
            'val_categories': val_categories
        }

        with open(self.output()['labels'].path, 'wb') as f:
            pickle.dump(labels, f)




