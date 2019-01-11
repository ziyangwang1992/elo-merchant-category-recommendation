# coding=utf-8
import sys

import pandas as pd

from config.config import *
from data.data_processor import *
from feature.base_extractor import *
from feature.transaction_extractor import TransactionExtractor
from feature.merchant_extractor import MerchantExtractor
from feature.trans_merchant_extractor import TransMerchantExtractor
from feature.card_extractor import CardExtractor


def extract_features():
    # 对商品进行初步处理
    mer_df = read_data(merchant_feature_input_path)
    merchant_extractor = MerchantExtractor(mer_df, merchant_feature_output_path)
    if merchant_extractor.extract_features() is None:
        sys.stderr.write("[FATAL]merchant_extractor result is None.\n")
        return False

    # 历史交易和商品按照merchant_id merge，求特征
    # 对历史交易进行初步处理
    his_tran_df = read_data(tran_feature_input_path)
    tran_extractor = TransactionExtractor(his_tran_df, tran_feature_output_path)
    if tran_extractor.extract_features() is None:
        sys.stderr.write("[FATAL]tran_extractor result is None.\n")
        return False

    # merge之后进行详细特征抽取
    merge_tran_mer_data = pd.merge(mer_df, his_tran_df, on='merchant_id', how='inner')
    tran_mer_extractor = TransMerchantExtractor(merge_tran_mer_data, trans_user_feature_output_path)
    if tran_mer_extractor.extract_features() is None:
        sys.stderr.write("[FATAL]tran_mer_extractor result is None.\n")
        return False



    card_feature_data = read_data(card_feature_input_path)
    card_extractor = CardExtractor(card_feature_data, card_feature_output_path)
    if card_extractor.get_result() is None:
        sys.stderr.write("[FATAL]card_extractor result is None.\n")
        return False
    return True

