import os

# col name
CARD_ID = 'card_id'
MERCHANT_ID = 'merchant_id'
AUTHORIZED_FLAG = 'authorized_flag'
INTERVAL_DAYS = 'interval_days'

# feature
# input/output path
his_tran_input_path = "../dataset/data/historical_transactions.csv"
his_tran_output_path = "../dataset/data/output/history.out.csv"
new_tran_input_path = "../dataset/data/new_merchant_transactions.csv"
new_tran_output_path = "../dataset/data/output/new.out.csv"
mer_input_path = "../dataset/data/merchants.csv"
mer_his_output_path = "../dataset/data/output/merchants.his.out.csv"
mer_new_output_path = "../dataset/data/output/merchants.new.out.csv"
train_card_input_path = "../dataset/data/train.csv"
card_feat_from_train_path = "../dataset/data/output/train.out.csv"
test_card_input_path = "../dataset/data/test.csv"
card_feat_from_test_path = "../dataset/data/output/test.out.csv"

his_tran_mer_path = "../dataset/data/output/his.tran.mer.csv"
card_feat_from_his_tran_mer_path = "../dataset/data/output/card.feature.his.csv"
new_tran_mer_path = "../dataset/data/output/new.tran.mer.csv"
card_feat_from_new_tran_mer_path = "../dataset/data/output/card.feature.new.csv"
card_feat_from_tran_mer_path = "../dataset/data/output/card.feature.tran.csv"

card_feature_path = "../dataset/data/output/card.feature.csv"
card_feature_predict_path = "../dataset/data/output/card.feature.predict.csv"

predict_path = "../dataset/data/output/predict.csv"
origin_result_path = "../dataset/data/sample_submission.csv"
result_path = "../dataset/data/output/result.csv"
