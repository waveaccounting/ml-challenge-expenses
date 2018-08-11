import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import os

# global variable that contains all the stop words (articles, prepositions and ...)
stop_words = set(stopwords.words('english'))


class DataFrame:
    def __init__(self, data_file, employee_file):
        self.data_file = data_file
        self.employee_file = employee_file

    @staticmethod
    def parse_csv(csv_file):
        """
        parse csv file and shuffle it if the specified phase is train
        :return: data frame containing csv file data
        """
        transaction_dataframe = pd.read_csv(csv_file)
        return transaction_dataframe

    @property
    def merge_data_frames(self):
        transaction_data_frame = self.parse_csv(self.data_file)
        employee_data_frame = self.parse_csv(self.employee_file)
        merged_data_frame = pd.merge(transaction_data_frame, employee_data_frame, how='inner', on="employee id")
        return merged_data_frame


class PrepareDataset(DataFrame):
    def __init__(self, data_file, employee_file, phase="Train"):
        """
        :param csv_file: train or validation csv file
        :param phase: train or validation (to shuffle data only for train phase)
        """
        DataFrame.__init__(self, data_file, employee_file)
        self.phase = phase
        self.transaction_dataframe = self.merge_data_frames

    @staticmethod
    def one_hot(column, prefix=None):
        """
        create one hot encoding for discrete or string labels/feature
        :param column: specified vector to be encoded
        :param prefix: a name for column in data frame
        :return: one-hot encoded label/feature
        """
        return pd.get_dummies(column, prefix=prefix)

    @property
    def word2vec(self):
        """
        the main purpose of this function is removing all the articles or prepositions from description feature
        and merge those samples that can have the same kind of feature. For instance, there are cases with descriptions
        like "Dinner", "Dinner with client", "Dinner with potential client". Using this function, value of "Dinner" will
        be assigned to all of these cases.
        It helps to have smaller dictionary for creating one-hot encoded vector for description.
        :return: simplified version of description column.
        """
        global stop_words
        descriptions = self.transaction_dataframe["expense description"]
        split_description = []
        for sentence in descriptions:
            words = sentence.split()
            # to remove articles and prepositions by using NLTK library
            split_description.append([word.lower() for word in words if word not in stop_words])

        # to keep track of the samples in which the description has been changed by making is_index_evaluated True
        # in those indices
        is_index_evaluated = [False] * len(split_description)
        new_description = [""] * len(split_description)

        for i in range(len(split_description)):
            # get the sub list
            sub_list = split_description[i]
            common_elements = []
            idx = [i]
            # check if it is already evaluated or not
            if not is_index_evaluated[i]:
                # evaluate the description of the remaining samples
                for j in range(i, len(split_description)):
                    compare_list = split_description[j]
                    # see if there is any common word between two sub lists
                    common = list(set(sub_list).intersection(compare_list))
                    # grab the index of the sample in case of having common word
                    if common:
                        common_elements.append(common)
                        idx.append(j)
                # grab the common word with smallest size. For instance,
                label = min(common_elements, key=len)
                # change the values and keep track of changed indices
                for id in idx:
                    new_description[id] = " ".join(label)
                    is_index_evaluated[id] = True

        return new_description

    def create_features(self):
        """
        create feature data frame from existing features and also add a new feature of tax rate
        :return: features
        """
        selected_features = pd.DataFrame()
        selected_features["pre-tax amount"] = self.transaction_dataframe["pre-tax amount"]
        selected_features["tax amount"] = self.transaction_dataframe["tax amount"]
        # create a synthetic feature of tax ratio
        selected_features["tax ratio"] = round(self.transaction_dataframe["tax amount"] / self.transaction_dataframe["pre-tax amount"],2)
        selected_features = pd.concat([selected_features, self.one_hot(self.transaction_dataframe["tax name"], prefix="tax name")], axis=1)
        selected_features = pd.concat([selected_features, self.one_hot(self.transaction_dataframe["employee id"], prefix="employee id")], axis=1)
        selected_features = pd.concat([selected_features, self.one_hot(self.word2vec, prefix="description")], axis=1)
        return selected_features

    def create_targets(self):
        """
        create one-hot encoded labels for each sample
        :return: labels
        """
        output_targets = self.transaction_dataframe["category"]
        return output_targets
