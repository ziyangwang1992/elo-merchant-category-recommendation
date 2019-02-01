import abc

import pandas as pd

from utils.data_processor import *


class BaseExtractor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_path, out_path, load=True, save=True):
        self._load = load
        self._save = save
        print("init input begin")
        if self._load:
            self._data = read_data(input_path)
        """
        f = open(input_path)
        reader = pd.read_csv(f, sep=',', iterator=True)
        loop = True
        chunkSize = 1000000
        chunks = []
        index = 0
        while loop:
            print("index: %d" % index)
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
            index += 1
        df = pd.concat(chunks, ignore_index=True)
        self._data = df
        """

        print("init input end")
        self._out_path = out_path

    @abc.abstractmethod
    def extract_features(self):
        pass

    def set_data(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def out_data(self):
        if not self._save:
            return True
        row, col = self._data.shape
        if row < 2 or col < 2:
            return False
        else:
            self._data.to_csv(self._out_path, header=True, index=False)
            return True
