from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import classification_report

import luigi
from luigi import LocalTarget
import pandas as pd
import pickle
import os
import numpy as np

from pre_processor import PreProcess


class Predictor(luigi.Task):
    """
    This class does the actual prediction and outputs a couple of files
    comparison.csv: A file showing actual and predicted categries side by side
    report.txt: Report on accuracy, precision and recall
    """

    output_path = luigi.Parameter()
    input_path = luigi.Parameter()

    def requires(self):
        return PreProcess(input_path=self.input_path,
                          output_path=self.output_path)

    def output(self):
        return {
            'result_report': LocalTarget(os.path.join(self.output_path, 'report.txt')),
            'result_comparison': LocalTarget(os.path.join(self.output_path, 'comparison.csv'))
        }

    def evaluate(self, model, x_train, y_train, x_val, y_val, cross_validate=False, grid_search_params={}):
        if cross_validate:
            grid_search_model = GridSearchCV(estimator=model,
                                             cv=LeaveOneOut(),
                                             param_grid=grid_search_params,
                                             verbose=0)
            grid_search_model.fit(x_train, y_train)
            return (grid_search_model.best_estimator_.score(x_train, y_train),
                    grid_search_model.best_estimator_.score(x_val, y_val),
                    grid_search_model.best_estimator_.predict(x_val),
                    grid_search_model.best_estimator_)
        else:
            model.fit(x_train, y_train)
            return model.score(x_train, y_train), model.score(x_val, y_val), model.predict(x_val), 0

    def run(self):

        # we won't use these 2 for predicting purpose here
        # we will only use the tf-idf features
        x_train = pd.read_csv(self.input()['train_x'].path)
        x_val = pd.read_csv(self.input()['validation_x'].path)

        x_train_tfidf = np.genfromtxt(self.input()['train_x_tfidf'].path, delimiter=',')
        x_val_tfidf = np.genfromtxt(self.input()['validation_x_tfidf'].path, delimiter=',')

        with open(self.input()['labels'].path, 'rb') as f:
            labels = pickle.load(f)

        y_train = labels['y_train']
        y_val = labels['y_val']
        categories = labels['categories']
        val_categories = labels['val_categories']

        grid_search_params = {'n_estimators': [10,25,50,100,200,300]}
        model = RandomForestClassifier(n_estimators=50, random_state=1)
        score_train, score_val, predictions, estimator = self.evaluate(model, x_train_tfidf,
                                                                       y_train, x_val_tfidf, y_val,
                                                                       cross_validate=True,
                                                                       grid_search_params=grid_search_params)

        actual_categories = [categories[i] for i in y_val]
        predicted_categories = [categories[i] for i in predictions]

        result_comp = pd.DataFrame({'actual': actual_categories,
                                    'prediction': predicted_categories})

        result_comp.to_csv(self.output()['result_comparison'].path, sep=',')

        with open(self.output()['result_report'].path, 'w') as f:
            f.write('Training Score: {} %\n'.format(score_train * 100))
            f.write('Validation Score: {} %\n\n'.format(score_val * 100))
            f.write(classification_report(y_val, predictions, target_names=val_categories))







