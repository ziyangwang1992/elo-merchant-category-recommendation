import sys

import pandas as pd

from config.config import *
from data.data_processor import *
from feature.base_extractor import *
from feature.transaction_extractor import TransactionExtractor
from feature.merchant_extractor import MerchantExtractor
from feature.card_extractor import CardExtractor


def extract_features():
    tran_feature_data = read_data(tran_feature_input_path)
    tran_extractor = TransactionExtractor(tran_feature_data, tran_feature_output_path)
    if tran_extractor.get_result() is None:
        sys.stderr.write("[FATAL]tran_extractor result is None.\n")
        return False

    merchant_feature_data = read_data(merchant_feature_input_path)
    merchant_extractor = MerchantExtractor(merchant_feature_data, merchant_feature_output_path)
    if merchant_extractor.get_result() is None:
        sys.stderr.write("[FATAL]merchant_extractor result is None.\n")
        return False

    card_feature_data = read_data(card_feature_input_path)
    card_extractor = CardExtractor(card_feature_data, card_feature_output_path)
    if card_extractor.get_result() is None:
        sys.stderr.write("[FATAL]card_extractor result is None.\n")
        return False
    return True

