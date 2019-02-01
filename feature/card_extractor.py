import sys

import pandas as pd

from utils.data_processor import *
from utils.time_processor import *
from feature.base_extractor import *
from config.config import *


class CardExtractor(BaseExtractor):
    def __init__(self, input_path, out_path):
        super(CardExtractor, self).__init__(input_path, out_path)

    def extract_features(self):
        df = self._data

        # one-hot
        feature_1 = pd.get_dummies(df['feature_1'], prefix='feature_1')
        df = pd.concat([df, feature_1], axis=1)

        feature_2 = pd.get_dummies(df['feature_2'], prefix='feature_2')
        df = pd.concat([df, feature_2], axis=1)

        feature_3 = pd.get_dummies(df['feature_3'], prefix='feature_3')
        df = pd.concat([df, feature_3], axis=1)

        com_date = "2018-05"
        df[INTERVAL_DAYS] = df['first_active_month'].apply(get_interval_days, str2=com_date)

        df.drop(['feature_1', 'feature_2', 'feature_3', 'first_active_month'], axis=1, inplace=True)

        # modify _data
        self.set_data(df)
        return self.out_data()
