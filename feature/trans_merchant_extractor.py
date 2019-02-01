#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

import pandas as pd

from config.config import *
from utils.data_processor import *
from utils.time_processor import *
from feature.base_extractor import *


class TransMerchantExtractor(BaseExtractor):
    def __init__(self, input_path, out_path, load=True, save=True, pre="his_"):
        super(TransMerchantExtractor, self).__init__(input_path, out_path, load=load, save=save)
        self._pre = pre

    # 用户交易次数
    def card_all_tran_times(self, df, group):
        print("--- card_all_tran_times ---")
        all_tran_times = group.size()
        all_tran_times.name = 'card_all_tran_times'
        return all_tran_times.reset_index()

    def feature_helper(self, df, group, key):
        result = df[[CARD_ID]].drop_duplicates()
        feat_list = list(self._data)
        for feat in feat_list:
            if key in feat:
                feat_times_name = feat + "_times"
                feat_time = group[feat].sum()
                feat_time.name = feat_times_name
                result = result.merge(feat_time.reset_index(), on=CARD_ID, how='left')

                feat_rate_name = feat + "_rate"
                feat_rate = group[feat].mean()
                feat_rate.name = feat_rate_name
                result = result.merge(feat_rate.reset_index(), on=CARD_ID, how='left')
        return result

    # authorized_flag
    def card_authorized_flag(self, df, group):
        return self.feature_helper(df, group, AUTHORIZED_FLAG)

    # month_lag
    def card_month_lag(self, df, group):
        return self.feature_helper(df, group, "month_lag")


    # category_3
    def card_category_3(self, df, group):
        return self.feature_helper(df, group, "category_3")

    # installment
    def card_installment(self, df, group):
        return self.feature_helper(df, group, "installment")

    # subsector,42
    def card_subsector(self, df, group):
        return self.feature_helper(df, group, "subsector")

    # category_1
    def card_category_1(self, df, group):
        return self.feature_helper(df, group, "category_1")

    # most_recent_sales_range
    def card_most_recent_sales_range(self, df, group):
        return self.feature_helper(df, group, "most_recent_sales_range")

    # most_most_recent_purchases_range
    def card_most_recent_purchases_range(self, df, group):
        return self.feature_helper(df, group, "most_most_recent_purchases_range")

    # active_months_lag3
    def card_active_months_lag3(self, df, group):
        return self.feature_helper(df, group, "active_months_lag3")

    # active_months_lag6
    def card_active_months_lag6(self, df, group):
        return self.feature_helper(df, group, "active_months_lag6")

    # active_months_lag12
    def card_active_months_lag12(self, df, group):
        return self.feature_helper(df, group, "active_months_lag12")

    # category_4
    def card_category_4(self, df, group):
        return self.feature_helper(df, group, "category_4")

    # state
    def card_state(self, df, group):
        return self.feature_helper(df, group, "state")

    # category_2
    def card_category_2(self, df, group):
        return self.feature_helper(df, group, "category_2")

    def extract_features(self):
        print("get_card begin")
        card_feature = self._data[[CARD_ID]].drop_duplicates()
        print("get_card end")
        df = self._data
        print("get_group begin")
        card_group = df.groupby(CARD_ID)
        print("get_group end")

        funcs = [self.card_all_tran_times, self.card_authorized_flag, self.card_month_lag,
                 self.card_category_3, self.card_installment,
                 self.card_subsector, self.card_category_1,
                 self.card_most_recent_sales_range, self.card_most_recent_purchases_range,
                 self.card_active_months_lag3, self.card_active_months_lag6,
                 self.card_active_months_lag12, self.card_category_4,
                 self.card_state, self.card_category_2]

        for f in funcs:
            card_feature = card_feature.merge(f(df, card_group), on=CARD_ID, how='left')

        # modify _data
        self.set_data(card_feature)

        if self:
            return self.out_data()
        return True
