import sys

import pandas as pd

from utils.data_processor import *
from feature.base_extractor import *


class TransMerchantExtractor(BaseExtractor):
    def __init__(self, input_path, out_path):
        super(TransMerchantExtractor, self).__init__(input_path, out_path)

    def extract_features(self):
        group = self._data.groupby('authorized_flag')
        user_feature = group['authorized_flag'].size()
        return user_feature
