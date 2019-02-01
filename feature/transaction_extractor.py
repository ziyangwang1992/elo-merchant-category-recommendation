import sys

import pandas as pd

from utils.data_processor import *
from feature.base_extractor import *
from config.config import *


class TransactionExtractor(BaseExtractor):
    def __init__(self, input_path, out_path, pre="his_"):
        super(TransactionExtractor, self).__init__(input_path, out_path)
        self._pre = pre

    def extract_features(self):
        if self._pre != "his_" and self._pre != "new_":
            sys.stderr.write("[FATAL]parameter pre is false, it must be his_ or new_.\n")
            return False

        df = self._data

        # one-hot
        month_lag = pd.get_dummies(df['month_lag'], prefix=self._pre+'month_lag')
        df = pd.concat([df, month_lag], axis=1)

        authorized_flag = pd.get_dummies(df['authorized_flag'], prefix=self._pre+'authorized_flag')
        df = pd.concat([df, authorized_flag], axis=1)

        category_3 = pd.get_dummies(df['category_3'], prefix=self._pre+'category_3')
        df = pd.concat([df, category_3], axis=1)

        installments = pd.get_dummies(df['installments'], prefix=self._pre+'installments')
        df = pd.concat([df, installments], axis=1)

        df = df.rename(columns={'purchase_amount': self._pre+'purchase_amount'})

        df.drop(["month_lag", "purchase_date", "authorized_flag", "category_3",
                 "installments", "category_1", "merchant_category_id",
                 "subsector_id", "city_id", "state_id", "category_2"], axis=1, inplace=True)

        # modify _data
        self.set_data(df)

        if "new_" == self._pre:
            return self.out_data()

        else:
            row, col = self._data.shape
            if row < 2 or col < 2:
                return False
            else:
                url_list = list(df[CARD_ID].drop_duplicates())
                all_url_list = list(df[CARD_ID])
                # print(url_list)
                url_size = len(url_list)

                step = int(url_size / 10)
                for i in range(1, 11):
                    print("out index: %d" % i)
                    begin = step*(i-1)
                    end = step*i
                    # print("begin: %d, end: %d" % (begin, end))
                    if i == 10:
                        urls = set(url_list[begin:])
                    else:
                        urls = set(url_list[begin: end])

                    url_mask = []
                    for url in all_url_list:
                        if url in urls:
                            url_mask.append(True)
                        else:
                            url_mask.append(False)

                    temp_df = df[url_mask]
                    temp_df.to_csv(self._out_path + "_" + str(i), header=True, index=False)

        return True
