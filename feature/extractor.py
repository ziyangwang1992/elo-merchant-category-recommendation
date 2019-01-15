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
    merchant_extractor = MerchantExtractor(mer_input_path, mer_output_path)
    if merchant_extractor.extract_features() is None:
        sys.stderr.write("[FATAL]merchant_extractor failed.\n")
        return False

    # 对历史交易进行初步处理
    his_tran_extractor = TransactionExtractor(his_tran_input_path, his_tran_output_path)
    if his_tran_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]history_transaction_extractor failed.\n")
        return False

    # 对新交易进行初步处理
    new_tran_extractor = TransactionExtractor(new_tran_input_path, new_tran_output_path)
    if new_tran_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]new_transaction_extractor failed.\n")
        return False

    # 对训练集和测试集的信用卡特征进行处理，得到信用卡特征1.1和1.2
    # train
    card_extractor = CardExtractor(train_card_input_path, card_feat_from_train_path)
    if card_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]train_card_extractor failed.\n")
        return False
    # test
    card_extractor = CardExtractor(test_card_input_path, card_feat_from_train_path)
    if card_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]test_card_extractor failed.\n")
        return False

    # 历史交易和商品按照merchant_id merge，进而得到信用卡特征2.1
    # 历史交易和商品特征合并
    his_tran_df = read_data(his_tran_output_path)
    mer_df = read_data(mer_output_path)
    his_tran_mer_df = pd.merge(his_tran_df, mer_df, on=MERCHANT_ID, how='inner')
    his_tran_mer_df.to_csv(his_tran_mer_path, header=True, index=False)
    # 分析合并表，得到信用卡特征
    his_tran_mer_extractor = TransMerchantExtractor(his_tran_mer_path, card_feat_from_his_tran_mer_path)
    if his_tran_mer_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]his_trans_merchant_card_extractor failed.\n")
        return False

    # 新交易和商品按照merchant_id merge，进而得到信用卡特征2.2
    new_tran_df = read_data(new_tran_output_path)
    mer_df = read_data(mer_output_path)
    new_tran_mer_df = pd.merge(new_tran_df, mer_df, on=MERCHANT_ID, how='inner')
    new_tran_mer_df.to_csv(new_tran_mer_path, header=True, index=False)
    # 分析合并表，得到信用卡特征
    new_tran_mer_extractor = TransMerchantExtractor(new_tran_mer_path, card_feat_from_new_tran_mer_path)
    if new_tran_mer_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]new_trans_merchant_card_extractor failed.\n")
        return False

    # merge 信用卡特征2.1和2.2，得到信用卡特征2
    card_feat_2_1 = read_data(card_feat_from_his_tran_mer_path)
    card_feat_2_2 = read_data(card_feat_from_new_tran_mer_path)
    card_feat_2 = pd.merge(card_feat_2_2, card_feat_2_1, on=CARD_ID, how='left')
    card_feat_2.to_csv(card_feat_from_tran_mer_path, header=True, index=False)

    # merge 信用卡特征1.1和2，得到最终用于训练的信用卡特征。1.2用于后边的预测
    card_feat_1 = read_data(card_feat_from_train_path)
    card_feat_2 = read_data(card_feat_from_tran_mer_path)
    card_feat = pd.merge(card_feat_1, card_feat_2, on=CARD_ID, how='left')
    card_feat.to_csv(card_feature_path, header=True, index=False)

    return True

