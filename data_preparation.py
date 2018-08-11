import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import os

# global variable that contains all the stop words (articles, prepositions and ...)
stop_words = set(stopwords.words('english'))


class Data:
    def __init__(self, csv_file, phase="Train"):
        """
        :param csv_file: train or validation csv file
        :param phase: train or validation (to shuffle data only for train phase)
        """
        self.csv_file = csv_file
        self.phase = phase
        self.transaction_dataframe = self.parse_csv

    @property
    def csv_file(self):
        """
        getter for csv_file
        :return path to the csv file
        """
        return self._csv_file

    @csv_file.setter
    def csv_file(self, value):
        """
        setter for csv_file (assign value to csv_file)
        :param value: a csv file
        """
        assert os.path.exists(os.path.normpath(value)), "csv file not found"
        self._csv_file = os.path.normpath(value)

    @property
    def phase(self):
        """
        getter for phase
        :return specified phase
        """
        return self._phase

    @phase.setter
    def phase(self, value):
        """
        setter for phase (assign value to phase)
        :param value: specified phase (train or validation)
        """
        assert value.lower() in ["train", "validation"]
        self._phase = value.lower()

    @property
    def parse_csv(self):
        """
        parse csv file and shuffle it if the specified phase is train
        :return: data frame containing csv file data
        """
        transaction_dataframe = pd.read_csv(self.csv_file)
        if self.phase == "train":
            np.random.seed(seed=0)
            sample_size = transaction_dataframe.shape[0]
            transaction_dataframe = transaction_dataframe.reindex(np.random.permutation(range(sample_size)))
        return transaction_dataframe

    @staticmethod
    def one_hot(column, prefix=None):
        """
        create one hot encoding for discrete or string labels/feature
        :param column: specified vector to be encoded
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
        selected_features["tax ratio"] = selected_features["tax amount"] / selected_features["pre-tax amount"]
        selected_features = pd.concat(
            [selected_features, self.one_hot(self.transaction_dataframe["tax name"], prefix="tax name")], axis=1)
        # selected_features = pd.concat([selected_features, self.one_hot(self.word2vec, prefix="description")], axis=1)
        return selected_features

    def create_targets(self):
        """
        create one-hot encoded labels for each sample
        :return: labels
        """
        output_targets = self.one_hot(self.transaction_dataframe["category"], prefix="description")
        return output_targets
