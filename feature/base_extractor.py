import abc

import pandas as pd

from data.data_processor import *


class BaseExtractor(metaclass=abc.ABCMeta):
    def __init__(self, input_path, out_path):
        self._data = read_data(input_path)
        self._out_path = out_path

    @abc.abstractmethod
    def extract_features(self):
        pass

