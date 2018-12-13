import sys

import pandas as pd

from data.data_processor import *
from feature.base_extractor import *


class MerchantExtractor(BaseExtractor):
    def __init__(self, data, out_path):
        super(MerchantExtractor, self).__init__(data, out_path)

    def extract_features(self):
        pass
