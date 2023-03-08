# STHAN-SR
复现AAAI-21-STHAN-SR（股票以及股票图关系数据处理见jupyter文件）


所有数据见：https://github.com/fulifeng/Temporal_Relational_Stock_Ranking

## train_NASDAQ_.py 以原论文方式复现
（以全部样本区间收盘价最大值标准化数据）
 + 2013-2015作为训练集，2016验证集，2017测试集
 

## train_NASDAQ2.py 时序交叉验证复现
 分四折时序交叉：（每折以训练集收盘价最大值标准化数据）
 + 2013训练 2014验证
 + 2013-2014训练 2015验证
 + 2013-2015训练 2016验证
 + 2013-2016训练 2017验证
