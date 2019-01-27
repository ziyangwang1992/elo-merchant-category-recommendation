import sys

import pandas as pd

from config.config import *
from utils.data_processor import *
from utils.time_processor import *
from feature.base_extractor import *


class TransMerchantExtractor(BaseExtractor):
    def __init__(self, input_path, out_path, pre="his_"):
        super(TransMerchantExtractor, self).__init__(input_path, out_path)
        self._pre = pre

    def user_all_tran_times(self, df, group):
        print("--- user_all_tran_times ---")
        all_tran_times = group.size()
        all_tran_times.name = 'user_all_tran_times'
        return all_tran_times.reset_index()

    def user_succ_tran_times(self, df, group):
        print("--- user_real_tran_times ---")
        name_authorized = self._pre + AUTHORIZED_FLAG
        print("name: %s" % name_authorized)
        user_real_buy = group[name_authorized + '_Y'].sum()
        user_real_buy.name = 'user_succ_tran_times'
        return user_real_buy.reset_index()

    def user_fail_train_times(self, df, group):
        print("--- user_fail_train_times ---")
        name_authorized = self._pre + AUTHORIZED_FLAG
        print("name: %s" % name_authorized)
        user_not_buy = group[name_authorized + '_N'].sum()
        user_not_buy.name = 'user_fail_train_times'
        return user_not_buy.reset_index()

    def extract_features(self):
        feat_list = list(self._data)
        user_feature = self._data[[CARD_ID]].drop_duplicates()
        df = self._data

        user_group = df.groupby(CARD_ID)
        funcs = [self.user_succ_tran_times, self.user_fail_train_times, self.user_all_tran_times]
        for f in funcs:
            user_feature = user_feature.merge(f(df, user_group), on=CARD_ID, how='left')

        # modify _data
        self.set_data(user_feature)
        return self.out_data()
