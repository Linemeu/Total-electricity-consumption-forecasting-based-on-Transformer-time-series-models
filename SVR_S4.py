from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import xgboost
from xgboost import XGBRegressor

data_month_ele = pd.read_excel(".\全国月度用电量_200801至202108_from万得数据库.xlsx", sheet_name="Sheet1")
data_month_exo = pd.read_excel(".\全国月度用电量_200801至202108_from万得数据库.xlsx", sheet_name="Sheet2")
data_month_ele["date"] = pd.to_datetime(data_month_ele["date"])
data_month_ele = data_month_ele.loc[data_month_ele["date"]>"2008-12-31", :]
data_month_ele.reset_index(inplace=True, drop=True)
data_month = pd.merge(data_month_ele, data_month_exo, on="date", how="left")
data_month=data_month.loc[data_month["date"]<"2018-05-31", :]
data_month.dropna(axis=1, how="any", inplace=True)
#window=12,143-window=143-12=131
# data_month.iloc[0:12]
data_month['month']=[data_month["date"][k].month for k in range(len(data_month))]
data_month_train=data_month.loc[data_month["date"]<"2017-01-31", :]
data_month_test=data_month.loc[data_month["date"]>="2017-01-31", :]
data_month_test=pd.concat([data_month_train[-12:],data_month_test])
data_month_train=data_month_train.iloc[:,1:]
data_month_test=data_month_test.iloc[:,1:]

#归一化
data_month_train_normelized=(data_month_train-data_month_train.mean())/data_month_train.std()
data_month_test_normelized=(data_month_test-data_month_train.mean())/data_month_train.std()
#mean:4409.034793//std:862.556798
train_mean=data_month_train.mean()[0]
train_std=data_month_train.std()[0]
def inverse_normalized(x,train_mean,train_std):
    # std=862.556798
    # mean=4409.034793
    x=x*train_std+train_mean
    return x
def data_gerenete(data_month):
    data_bag = []
    for i in range(0, len(data_month) - 12):
        data_bag.append(data_month.iloc[i:13 + i])

    dec_data = [np.array(data_bag[k].iloc[:, 0][:-1]) for k in range(0, len(data_bag))]
    dec_data = np.array(dec_data).reshape(len(data_month) - 12, 12, 1)

    target_data = [np.array(data_bag[k].iloc[:, 0][-1:]) for k in range(0, len(data_bag))]
    target_data = np.array(target_data).reshape(len(data_month) - 12, 1, 1)

    enc_data = [np.array(data_bag[k].iloc[:, 1:-1][:-1]) for k in range(0, len(data_bag))]
    enc_data =np.array(enc_data).reshape(len(data_month) - 12, 12, 27)

    enc_mark= [np.array(data_bag[k].iloc[:, -1][:-1]) for k in range(0, len(data_bag))]
    dec_mark = [np.array(data_bag[k].iloc[:, -1][:-1]) for k in range(0, len(data_bag))]
    enc_mark = np.array(enc_mark).reshape(len(data_month)-12, 12, 1)
    dec_mark = np.array(dec_mark).reshape(len(data_month)-12, 12, 1)

    return enc_data, dec_data, target_data,enc_mark,dec_mark
#计算测试集的mse，mape，mae
def mse(true, predict):
    ture=np.array(true)
    predict=np.array(predict)
    return np.mean(np.power(true-predict, 2))
def mae(true, predict):
    ture=np.array(true)
    predict=np.array(predict)
    return np.mean(np.abs(true-predict))
def mape(true, predict):
    ture=np.array(true)
    predict=np.array(predict)
    return np.mean(np.abs(true-predict)/np.abs(true))
#训练集数据
train_data_enc, train_data_dec, train_data_out,enc_mark,dec_mark=data_gerenete(data_month_train_normelized)
train_data_enc=train_data_enc
train_data_dec=train_data_dec
train_data_out=train_data_out
enc_mark=enc_mark
dec_mark=dec_mark
train_length=enc_mark.shape[0]
train_data_input=np.concatenate([train_data_enc,train_data_dec,enc_mark],axis=2).reshape([train_length,-1])
train_data_output=train_data_out.reshape([train_length,-1])
#测试及数据构建
test_data_enc, test_data_dec, test_data_out,enc_mark,dec_mark=data_gerenete(data_month_test_normelized)
test_length=test_data_dec.shape[0]
test_data_input=np.concatenate([test_data_enc,test_data_dec,enc_mark],axis=2).reshape([test_length,-1])
test_data_output=test_data_out.reshape([test_length,-1])
# class Config(object):
#     # def __init__(self):
#     hidden_dim=32
#     batch_size=1  # batch size
#     num_layers=2
#     feature_num=9
#     epoch=1000
#     keep_dropout=0.1
#     num_classes=1
#     train_window=12
#     lr = 0.0001  # initial learning rate
#     lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
#     weight_decay = 0.0001  # 损失函数
# model = SVRRegressor( hidden_layer_sizes=(54,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
#         learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
#         random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#         early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = SVR()
# model = SVRRegressor( )
model.fit(train_data_input, train_data_output)
pred_out=[]
for i in range(len(test_data_out)):
    out=model.predict(test_data_input[i].reshape(1,-1))
    pred_out.append(out.tolist())

inv_predict_squence=[inverse_normalized(k[0],train_mean,train_std) for k in pred_out]
inv_truth_squence=[inverse_normalized(k[0],train_mean,train_std) for k in test_data_output]
mse(inv_truth_squence, inv_predict_squence)
mape(inv_truth_squence, inv_predict_squence)
mae(inv_truth_squence, inv_predict_squence)
inv_predict_squence = [k for k in inv_predict_squence]
inv_truth_squence = [k for k in inv_truth_squence]
mse(inv_truth_squence, inv_predict_squence)
mape(inv_truth_squence, inv_predict_squence)
mae(inv_truth_squence, inv_predict_squence)
import json
S1_result = [mape(inv_truth_squence, inv_predict_squence),
             mse(inv_truth_squence, inv_predict_squence),
             mae(inv_truth_squence, inv_predict_squence)]
import pandas as pd
try:
    truth_predict_data=pd.read_excel('.\\truth_predict_data_S4.xlsx')
except:
    print('没找到文件')
    truth_predict_data=pd.DataFrame()
truth_predict_data['S4_SVR_truth']=inv_truth_squence
truth_predict_data['S4_SVR_predict']=inv_predict_squence
truth_predict_data.to_excel('truth_predict_data_S4.xlsx')

with open('result_dict.json', 'r') as json_file:
    result_dict = json.load(json_file)
result_dict['S4_SVR_mape'] = float(S1_result[0])
result_dict['S4_SVR_mse'] = float(S1_result[1])
result_dict['S4_SVR_mae'] = float(S1_result[2])
# mape,mse,mae
dict_json = json.dumps(result_dict)
with open('result_dict.json', 'w+') as file:
    file.write(dict_json)
result_dict