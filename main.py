import numpy as np
import argparse
import sys
import os

from data_preparation import *
from classifiers import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_train', type=str, help='input train data csv file path')
    parser.add_argument('csv_validation', type=str, help='input validation data csv file path')
    parser.add_argument('csv_employee', type=str, help='input employee data csv file path')

    try:
        args = parser.parse_args()
        assert os.path.exists(os.path.normpath(args.csv_train)), "train data not found"
        assert os.path.exists(os.path.normpath(args.csv_validation)), "validation data not found"
        assert os.path.exists(os.path.normpath(args.csv_employee)), "employee data not found"
        training_data_example = os.path.normpath(args.csv_train)
        validation_data_example = os.path.normpath(args.csv_validation)
        employee = os.path.normpath(args.csv_employee)

        train = PrepareDataset(training_data_example, employee)
        x_train = train.create_features()
        y_train_one_hot = train.create_targets()

        validation = PrepareDataset(validation_data_example, employee, phase="Validation")
        x_validation = validation.create_features()
        y_validation_one_hot = validation.create_targets()

        # align train and validation data-frames to fill the missing features and labels
        x_train, x_validation = x_train.align(x_validation, axis=1, fill_value=0)
        y_train_one_hot, y_validation_one_hot = y_train_one_hot.align(y_validation_one_hot, axis=1, fill_value=0)

        x_train, y_train_one_hot = x_train.values, y_train_one_hot.values
        x_validation, y_validation_one_hot = x_validation.values, y_validation_one_hot.values

        y_train = [np.where(r == 1)[0][0] for r in y_train_one_hot]
        y_validation = [np.where(r == 1)[0][0] for r in y_validation_one_hot]

        train = Train(x_train, y_train)
        models = {"KNN": train.knn, "Random Forest": train.random_forest, "SVM": train.svm}

        for key, value in models.items():
            acc_train = Validate(value, x_train, y_train).predict
            acc_validation = Validate(value, x_validation, y_validation).predict
            print("* Classification results of {0}: "
                  "\n\t> Training accuracy = {1:.2f} "
                  "\n\t> Validation accuracy = {2:.2f}\n".format(key, acc_train, acc_validation))

    except AssertionError as message:
        sys.exit(message)
