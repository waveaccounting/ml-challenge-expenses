import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import os

stop_words = set(stopwords.words('english'))


class Data:
    def __init__(self, csv_file, phase="Train"):
        self.csv_file = csv_file
        self.phase = phase
        self.transaction_dataframe = self.parse_csv

    @property
    def csv_file(self):
        return self._csv_file

    @csv_file.setter
    def csv_file(self, value):
        assert os.path.exists(os.path.normpath(value)), "csv file not found"
        self._csv_file = os.path.normpath(value)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        assert value.lower() in ["train", "validation"]
        self._phase = value.lower()

    @property
    def parse_csv(self):
        transaction_dataframe = pd.read_csv(self.csv_file)
        if self.phase == "train":
            np.random.seed(seed=0)
            sample_size = transaction_dataframe.shape[0]
            transaction_dataframe = transaction_dataframe.reindex(np.random.permutation(range(sample_size)))
        return transaction_dataframe

    @staticmethod
    def one_hot(column):
        return pd.get_dummies(column).values.tolist()

    @property
    def word2vec(self):
        global stop_words
        descriptions = self.transaction_dataframe["expense description"]
        split_description = []
        for sentence in descriptions:
            words = sentence.split()
            # to remove articles and prepositions by using NLTK library
            split_description.append([word.lower() for word in words if word not in stop_words])

        is_index_evaluated = [False] * len(split_description)
        new_description = [""] * len(split_description)

        for i in range(len(split_description)):
            sub_list = split_description[i]
            common_elements = []
            idx = [i]
            if not is_index_evaluated[i]:
                for j in range(i, len(split_description)):
                    compare_list = split_description[j]
                    common = list(set(sub_list).intersection(compare_list))
                    if common:
                        common_elements.append(common)
                        idx.append(j)

                label = min(common_elements, key=len)
                for id in idx:
                    new_description[id] = " ".join(label)
                    is_index_evaluated[id] = True

        return new_description

    def create_features(self):
        selected_features = pd.DataFrame()
        selected_features["employee id"] = self.one_hot(self.transaction_dataframe["employee id"])
        selected_features["pre-tax amount"] = self.transaction_dataframe["pre-tax amount"]
        selected_features["tax name"] = self.one_hot(self.transaction_dataframe["tax name"])
        selected_features["tax amount"] = self.transaction_dataframe["tax amount"]
        # create a synthetic feature of tax ratio
        selected_features["tax ratio"] = round(selected_features["tax amount"] / selected_features["pre-tax amount"], 2)
        selected_features["description"] = self.one_hot(self.word2vec)
        return selected_features

    def create_targets(self):
        output_targets = pd.DataFrame()
        output_targets["category"] = self.one_hot(self.transaction_dataframe["category"])
        return output_targets


if __name__ == "__main__":
    training_data_example = "./training_data_example.csv"
    train_data = Data(training_data_example)
    training_features = train_data.create_features()
    training_targets = train_data.create_targets()
    print(training_features)

