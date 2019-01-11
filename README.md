tips：
1. 历史交易和新交易分开处理特征
2. 先用交易表和商家表join，进而提取用户交易特征，再进一步提取用户行为特征，最后和train.csv中的用户特征join


******************
**** 交易特征 ****
******************
信用卡id，card_id
距参考日期的月份差距，month_lag
是否授权交易，authorized_flag
交易类别1
交易类别2
交易类别3
分期付款次数，installments
归一化的购买数量，purchase_amount
交易省
交易市
商家类别，328个值，merchant_category_id
商家类组，42个值，subsector_id
商家id，merchant_id

******************
**** 商品特征 ****
******************
商家id，merchant_id
category_1
most_recent_sales_range
most_recent_purchases_range
active_months_lag3
active_months_lag6
active_months_lag12
category_4
city_id
state_id
category_2

******************
*** 信用卡特征 ***
******************
发起交易次数
授权交易次数
未授权交易次数
授权率
授权交易商家数

交易覆盖省个数
交易覆盖市个数

