#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xgboost as xgb
import numpy as np

from config.config import *
from feature.base_extractor import *

np.set_printoptions(suppress=True)


def xgboost_fit():
    print("----- xgboost_fit begin -----")

    df = read_data(card_feature_path)
    df.drop([CARD_ID], axis=1, inplace=True)

    # 默认值为0
    df.fillna(0, inplace=True)

    target = df['target']
    df_train = df.drop(['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_train, target, test_size=0.3, random_state=1)

    # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    # model.fit(X_train, y_train)

    data_train = xgb.DMatrix(X_train, y_train)  # 使用XGBoost的原生版本需要对数据进行转化
    data_test = xgb.DMatrix(X_test, y_test)

    param = {'max_depth': 20, 'eta': 1, 'objective': 'reg:linear'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 20
    booster = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)

    # 计算错误率
    y_predicted = booster.predict(data_test)
    y = data_test.get_label()

    accuracy = sum(y == (y_predicted > 0.5))
    accuracy_rate = float(accuracy) / len(y_predicted)
    print('样本总数：{0}'.format(len(y_predicted)))
    print('正确数目：{0}'.format(accuracy))
    print('正确率：{0:.3f}'.format(accuracy_rate))

    print("----- xgboost_fit end -----")
    return booster


def xgboost_predict(xgb_model):
    print("----- xgboost_predict begin -----")
    test_df = read_data(card_feature_predict_path)
    test_df.drop([CARD_ID], axis=1, inplace=True)
    test_df.fillna(0, inplace=True)

    dtest = xgb.DMatrix(test_df)
    predict = xgb_model.predict(dtest)

    target = pd.Series(predict)
    target.name = "target"

    result_df = read_data(origin_result_path)
    result_df['target'] = target
    result_df.to_csv(result_path, header=True, index=False)

    print("----- xgboost_predict end -----")
    return True
