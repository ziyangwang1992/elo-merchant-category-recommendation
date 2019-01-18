import abc

import pandas as pd

from utils.data_processor import *


class BaseExtractor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_path, out_path):
        self._data = read_data(input_path)
        self._out_path = out_path

    @abc.abstractmethod
    def extract_features(self):
        pass

    def set_data(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def out_data(self):
        row, col = self._data.shape
        if row < 2 or col < 2:
            return False
        else:
            self._data.to_csv(self._out_path, header=True, index=False)
            return True
