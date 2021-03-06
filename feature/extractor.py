#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

import pandas as pd

from config.config import *
from utils.data_processor import *
from utils.time_processor import *
from feature.base_extractor import *
from feature.transaction_extractor import TransactionExtractor
from feature.merchant_extractor import MerchantExtractor
from feature.trans_merchant_extractor import TransMerchantExtractor
from feature.card_extractor import CardExtractor


def extract_features():
    print("----- extract_features begin -----")
    """
    # 对商品进行初步处理
    print("----- MerchantExtractor begin -----")
    merchant_extractor = MerchantExtractor(mer_input_path, mer_his_output_path, pre="his_")
    if merchant_extractor.extract_features() is None:
        sys.stderr.write("[FATAL]merchant_extractor failed.\n")
        return False
    merchant_extractor = MerchantExtractor(mer_input_path, mer_new_output_path, pre="new_")
    if merchant_extractor.extract_features() is None:
        sys.stderr.write("[FATAL]merchant_extractor failed.\n")
        return False
    print("----- MerchantExtractor end -----")


    # 对历史交易进行初步处理，并将结果切成十份
    print("----- History TransactionExtractor begin -----")
    his_tran_extractor = TransactionExtractor(his_tran_input_path, his_tran_output_path, pre="his_")
    if his_tran_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]history_transaction_extractor failed.\n")
        return False
    print("----- History TransactionExtractor end -----")

    # 对新交易进行初步处理
    print("----- New TransactionExtractor begin -----")
    new_tran_extractor = TransactionExtractor(new_tran_input_path, new_tran_output_path, pre="new_")
    if new_tran_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]new_transaction_extractor failed.\n")
        return False
    print("----- New TransactionExtractor end -----")

    # 对训练集和测试集的信用卡特征进行处理，得到信用卡特征1.1和1.2
    # train
    print("----- Train CardExtractor begin -----")
    card_extractor = CardExtractor(train_card_input_path, card_feat_from_train_path)
    if card_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]train_card_extractor failed.\n")
        return False
    print("----- Train CardExtractor end -----")
    # test
    print("----- Test CardExtractor begin -----")
    card_extractor = CardExtractor(test_card_input_path, card_feat_from_test_path)
    if card_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]test_card_extractor failed.\n")
        return False
    print("----- Test CardExtractor end -----")

    # 历史交易和商品按照merchant_id merge，进而得到信用卡特征2.1
    # 数据量太大，无法全部加载到内存，手动分成10个part运行，最后合并
    print("----- merge history transaction and merchant begin -----")
    chunks = []
    mer_df = read_data(mer_his_output_path)
    for i in range(1, 11):
        print("index: %d" % i)
        # 历史交易和商品特征合并
        print("--- merge history transaction and merchant ---")
        his_tran_df = read_data(his_tran_output_path + "_" + str(i))
        his_tran_mer_df = pd.merge(his_tran_df, mer_df, on=MERCHANT_ID, how='left')
        #his_tran_mer_df.to_csv(his_tran_mer_path + "_" + str(i), header=True, index=False)
        # 分析合并表，得到信用卡特征
        print("--- history transaction merchant Extractor ---")
        # his_tran_mer_extractor = TransMerchantExtractor(his_tran_mer_path + "_" + str(i),
        #                                                 card_feat_from_his_tran_mer_path + "_" + str(i),
        #                                                 pre="his_")
        his_tran_mer_extractor = TransMerchantExtractor("",
                                                        card_feat_from_his_tran_mer_path + "_" + str(i),
                                                        load=False,
                                                        save=False,
                                                        pre="his_")
        his_tran_mer_extractor.set_data(his_tran_mer_df)
        if his_tran_mer_extractor.extract_features() is False:
            sys.stderr.write("[FATAL]his_trans_merchant_card_extractor failed.\n")
            return False
        chunks.append(his_tran_mer_extractor.get_data())
    card_feat_from_his_df = pd.concat(chunks, ignore_index=True)
    card_feat_from_his_df.to_csv(card_feat_from_his_tran_mer_path, header=True, index=False)

    # 新交易和商品按照merchant_id merge，进而得到信用卡特征2.2
    print("----- merge new transaction and merchant begin -----")
    new_tran_df = read_data(new_tran_output_path)
    mer_df = read_data(mer_new_output_path)
    new_tran_mer_df = pd.merge(new_tran_df, mer_df, on=MERCHANT_ID, how='left')
    new_tran_mer_df.to_csv(new_tran_mer_path, header=True, index=False)
    # 分析合并表，得到信用卡特征
    print("----- new transaction merchant Extractor begin -----")
    new_tran_mer_extractor = TransMerchantExtractor(new_tran_mer_path,
                                                    card_feat_from_new_tran_mer_path, pre="new_")
    if new_tran_mer_extractor.extract_features() is False:
        sys.stderr.write("[FATAL]new_trans_merchant_card_extractor failed.\n")
        return False
    """

    # merge 信用卡特征2.1和2.2，得到信用卡特征2
    print("----- merge card feature from transaction begin -----")
    card_feat_2_1 = read_data(card_feat_from_his_tran_mer_path)
    card_feat_2_2 = read_data(card_feat_from_new_tran_mer_path)
    card_feat_2 = pd.merge(card_feat_2_2, card_feat_2_1, on=CARD_ID, how='left')
    card_feat_2.to_csv(card_feat_from_tran_mer_path, header=True, index=False)

    # merge 信用卡特征1.1和2，得到最终用于训练的信用卡特征。1.2用于后边的预测
    print("----- merge train card feature begin -----")
    card_feat_1 = read_data(card_feat_from_train_path)
    card_feat_2 = read_data(card_feat_from_tran_mer_path)
    card_feat = pd.merge(card_feat_1, card_feat_2, on=CARD_ID, how='left')
    card_feat.to_csv(card_feature_path, header=True, index=False)

    # merge 信用卡特征1.2和2，即merge测试集和信用卡特征，用于预测
    print("----- merge test feature begin -----")
    card_test_feature = read_data(card_feat_from_test_path)
    card_feature = read_data(card_feat_from_tran_mer_path)
    predict_df = pd.merge(card_test_feature, card_feature, on=CARD_ID, how='left')
    predict_df.to_csv(card_feature_predict_path, header=True, index=False)

    print("----- extract_features end -----")

    return True

