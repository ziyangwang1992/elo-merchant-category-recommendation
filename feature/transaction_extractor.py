import sys

import pandas as pd

from utils.data_processor import *
from feature.base_extractor import *


class TransactionExtractor(BaseExtractor):
    def __init__(self, input_path, out_path, pre="his_"):
        super(TransactionExtractor, self).__init__(input_path, out_path)
        self._pre = pre

    def extract_features(self):
        df = self._data

        # one-hot
        month_lag = pd.get_dummies(df['month_lag'], prefix='month_lag')
        df = pd.concat([df, month_lag], axis=1)

        authorized_flag = pd.get_dummies(df['authorized_flag'], prefix='authorized_flag')
        df = pd.concat([df, authorized_flag], axis=1)

        category_3 = pd.get_dummies(df['category_3'], prefix=self._pre+'category_3')
        df = pd.concat([df, category_3], axis=1)

        installments = pd.get_dummies(df['installments'], prefix='installments')
        df = pd.concat([df, installments], axis=1)

        # modify _data
        self.set_data(df)
        return self.out_data()
