import sys

import pandas as pd

from utils.data_processor import *
from feature.base_extractor import *


class MerchantExtractor(BaseExtractor):
    def __init__(self, input_path, out_path, pre="his_"):
        super(MerchantExtractor, self).__init__(input_path, out_path)
        self._pre = pre

    def extract_features(self):
        if self._pre != "his_" and self._pre != "new_":
            sys.stderr.write("[FATAL]parameter pre is false, it must be his_ or new_.\n")
            return False

        df = self._data

        # one-hot
        subsector_id = pd.get_dummies(df['subsector_id'], prefix=self._pre+'subsector_id')
        df = pd.concat([df, subsector_id], axis=1)

        category_1 = pd.get_dummies(df['category_1'], prefix=self._pre+'category_1')
        df = pd.concat([df, category_1], axis=1)

        most_recent_sales_range = pd.get_dummies(df['most_recent_sales_range'],
                                                 prefix=self._pre+'most_recent_sales_range')
        df = pd.concat([df, most_recent_sales_range], axis=1)

        most_recent_purchases_range = pd.get_dummies(df['most_recent_purchases_range'],
                                                     prefix=self._pre+'most_recent_purchases_range')
        df = pd.concat([df, most_recent_purchases_range], axis=1)

        active_months_lag3 = pd.get_dummies(df['active_months_lag3'], prefix=self._pre+'active_months_lag3')
        df = pd.concat([df, active_months_lag3], axis=1)

        active_months_lag6 = pd.get_dummies(df['active_months_lag6'], prefix=self._pre+'active_months_lag6')
        df = pd.concat([df, active_months_lag6], axis=1)

        active_months_lag12 = pd.get_dummies(df['active_months_lag12'], prefix=self._pre+'active_months_lag12')
        df = pd.concat([df, active_months_lag12], axis=1)

        category_4 = pd.get_dummies(df['category_4'], prefix=self._pre+'category_4')
        df = pd.concat([df, category_4], axis=1)

        #city_id = pd.get_dummies(df['city_id'], prefix=self._pre+'city_id')
        #df = pd.concat([df, city_id], axis=1)

        state_id = pd.get_dummies(df['state_id'], prefix=self._pre+'state_id')
        df = pd.concat([df, state_id], axis=1)

        category_2 = pd.get_dummies(df['category_2'], prefix=self._pre+'category_2')
        df = pd.concat([df, category_2], axis=1)

        df.drop(["subsector_id", "merchant_group_id", "merchant_category_id",
                 "subsector_id", "numerical_1", "numerical_2", "category_1",
                 "most_recent_sales_range", "most_recent_purchases_range",
                 "avg_sales_lag3", "avg_purchases_lag3", "active_months_lag3",
                 "avg_sales_lag6", "avg_purchases_lag6", "active_months_lag6",
                 "avg_sales_lag12", "avg_purchases_lag12", "active_months_lag12",
                 "category_4", "city_id", "state_id", "category_2"], axis=1, inplace=True)

        # modify _data
        self.set_data(df)
        return self.out_data()
