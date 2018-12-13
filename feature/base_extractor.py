import abc

import pandas as pd

from data.data_processor import *


class BaseExtractor(metaclass=abc.ABCMeta):
    def __init__(self, data, out_path):
        self._data = data
        self._result = None
        self._out_path = out_path

    @abc.abstractmethod
    def extract_features(self):
        pass

    def get_result(self):
        return self._result
