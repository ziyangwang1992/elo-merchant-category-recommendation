import sys

import pandas as pd

from utils.data_processor import *
from feature.base_extractor import *


class MerchantExtractor(BaseExtractor):
    def __init__(self, input_path, out_path):
        super(MerchantExtractor, self).__init__(input_path, out_path)

    def extract_features(self):
        df = self._data

        # one-hot
        subsector_id = pd.get_dummies(df['subsector_id'], prefix='subsector_id')
        df = pd.concat([df, subsector_id], axis=1)

        category_1 = pd.get_dummies(df['category_1'], prefix='category_1')
        df = pd.concat([df, category_1], axis=1)

        most_recent_sales_range = pd.get_dummies(df['most_recent_sales_range'],
                                                 prefix='most_recent_sales_range')
        df = pd.concat([df, most_recent_sales_range], axis=1)

        most_recent_purchases_range = pd.get_dummies(df['most_recent_purchases_range'],
                                                     prefix='most_recent_purchases_range')
        df = pd.concat([df, most_recent_purchases_range], axis=1)

        active_months_lag3 = pd.get_dummies(df['active_months_lag3'], prefix='active_months_lag3')
        df = pd.concat([df, active_months_lag3], axis=1)

        active_months_lag6 = pd.get_dummies(df['active_months_lag6'], prefix='active_months_lag6')
        df = pd.concat([df, active_months_lag6], axis=1)

        active_months_lag12 = pd.get_dummies(df['active_months_lag12'], prefix='active_months_lag12')
        df = pd.concat([df, active_months_lag12], axis=1)

        category_4 = pd.get_dummies(df['category_4'], prefix='category_4')
        df = pd.concat([df, category_4], axis=1)

        city_id = pd.get_dummies(df['city_id'], prefix='city_id')
        df = pd.concat([df, city_id], axis=1)

        state_id = pd.get_dummies(df['state_id'], prefix='state_id')
        df = pd.concat([df, state_id], axis=1)

        category_2 = pd.get_dummies(df['category_2'], prefix='category_2')
        df = pd.concat([df, category_2], axis=1)

        # modify _data
        self.set_data(df)
        return self.out_data()
