#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from utils.data_processor import *
from feature.extractor import extract_features
from models.xgboost_model import *
from config.config import *


if __name__ == '__main__':
    # gen features
    """
    result = extract_features()
    if result is False:
        sys.stderr.write("[FATAL]extract_features failed.\n")
        sys.exit(1)
    """

    # train model
    xgb_model = xgboost_fit()

    # predict
    xgboost_predict(xgb_model)

