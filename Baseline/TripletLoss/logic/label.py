import json

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .csvreader import read_labels


class Label:
    def __init__(self, labels):
        self.ohe = None
        self.le = None
        self.label_series = None

        self.labels = None
        self.label_count = None

        if isinstance(labels, str):
            self.label_path = labels
        else:
            self.label_path = None

        self.__read_labels__(labels)

    def __read_labels__(self, labels):
        if isinstance(labels, str):  # labels is path
            # labels_file = open(labels, 'r').readlines()
            # smells = []
            # for i in labels_file:
            #     test_line = json.loads(i)
            #     smells.append(test_line["smellKey"])
            #
            # self.label_series = pd.Series(smells)
            self.label_series = read_labels(labels)
        else:
            self.label_series = labels.copy()

        self.ohe = OneHotEncoder(sparse_output=False)
        self.le = LabelEncoder()
        self.labels = self.ohe.fit_transform(self.label_series.to_numpy().reshape(-1, 1))
        self.le.fit(self.label_series.to_numpy())
        self.label_count = len(self.ohe.categories_[0])
