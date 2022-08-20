import pandas as pd
import numpy as np


class DataSet:
    def __init__(self):
        self._data_table = pd.DataFrame()
        self._generated_indexes = []

    def get_df(self):
        return self._data_table

    def load_csv(self, path):
        self._data_table = pd.read_csv(path)
        if self._data_table.columns[0] == 'Unnamed: 0':
            self._data_table = self._data_table.drop(self._data_table.columns[0], axis=1)

    def init_by_values(self, column_names, values=None):
        columns = []
        if type(column_names) == np.ndarray:
            columns = column_names.tolist()
        elif type(column_names) == list:
            columns = column_names
        else:
            raise TypeError('Invalid type of columns names')

        if values:
            if type(values) == np.ndarray:
                values = values.tolist()
            elif type(values) != list:
                raise TypeError('Invalid type of values')

        self._data_table = pd.DataFrame(values, columns=columns)

    def init_by_df(self, df):
        if type(df) == pd.DataFrame:
            self._data_table = df.copy()

    def save_csv(self, path):
        self._data_table.to_csv(path)
        