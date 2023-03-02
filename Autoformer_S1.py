# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_addTimeEmbed
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_addTimeEmbed(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_addTimeEmbed(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # # decoder input
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # print('zeros.shape',zeros.shape)
        # print('seasonal_init[:, -self.label_len:, :].shape',seasonal_init[:, -self.label_len:, :].shape)
        # seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        # dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        seasonal_init_dec, trend_init_dec = self.decomp(x_dec)
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        # print('dec_out.shape', dec_out.shape)
        # print('seasonal_init', seasonal_init.shape)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init_dec)
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
        #                                          trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 12
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = 12
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 12
        label_len = 11
        pred_len =1
        output_attention = True
        enc_in = 2
        dec_in = 1
        #---------------------
        d_model = 64
        embed = 'timeF'
        dropout = 0.05
        freq = 'm'
        factor = 1
        n_heads = 1
        d_ff = 16
        e_layers = 1
        d_layers = 1
        c_out = 1
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs).to(device)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([1, configs.seq_len, configs.enc_in]).to(device)
    enc_mark = torch.randn([1, configs.seq_len, 1]).to(device)

    # dec = torch.randn([1, configs.seq_len//2+configs.pred_len, 1])
    dec = torch.randn([1, configs.seq_len, configs.dec_in]).to(device)
    # dec_mark = torch.randn([1, configs.seq_len//2+configs.pred_len, 4])
    dec_mark = torch.randn([1, configs.seq_len, 1]).to(device)
    print('dec_mark',dec_mark.shape)
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)


import pandas as pd
import numpy as np


data_month_ele = pd.read_excel(".\全国月度用电量_200801至202108_from万得数据库.xlsx", sheet_name="Sheet1")
data_month_exo = pd.read_excel(".\全国月度用电量_200801至202108_from万得数据库.xlsx", sheet_name="Sheet2")
data_month_ele["date"] = pd.to_datetime(data_month_ele["date"])
data_month_ele = data_month_ele.loc[data_month_ele["date"]>"2008-12-31", :]
data_month_ele = data_month_ele.loc[data_month_ele["date"]<="2011-04-30", :]
data_month_ele.reset_index(inplace=True, drop=True)
data_month = pd.merge(data_month_ele, data_month_exo, on="date", how="left")
data_month=data_month.loc[data_month["date"]<="2011-04-30", :]
data_month.dropna(axis=1, how="any", inplace=True)
#window=12,143-window=143-12=131
# data_month.iloc[0:12]
data_month['month']=[data_month["date"][k].month for k in range(len(data_month))]
data_month_train=data_month.loc[data_month["date"]<="2010-12-31", :]
data_month_test=data_month.loc[data_month["date"]>"2010-12-31", :]
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
    std=train_std
    mean=train_mean
    x=x*std+mean
    return x
def data_gerenete(data_month):
    data_bag = []
    for i in range(0, len(data_month) - 12):
        data_bag.append(data_month.iloc[i:13 + i])

    dec_data = [np.array(data_bag[k].iloc[:, 0][:-1]) for k in range(0, len(data_bag))]
    dec_data = torch.Tensor(dec_data).reshape(len(data_month) - 12, 12, 1)

    target_data = [np.array(data_bag[k].iloc[:, 0][-1:]) for k in range(0, len(data_bag))]
    target_data = torch.Tensor(target_data).reshape(len(data_month) - 12, 1, 1)

    enc_data = [np.array(data_bag[k].iloc[:, 1:-1][:-1]) for k in range(0, len(data_bag))]
    enc_data = torch.Tensor(enc_data).reshape(len(data_month) - 12, 12, 27)

    enc_mark= [np.array(data_bag[k].iloc[:, -1][:-1]) for k in range(0, len(data_bag))]
    dec_mark = [np.array(data_bag[k].iloc[:, -1][:-1]) for k in range(0, len(data_bag))]
    enc_mark = torch.Tensor(enc_mark).reshape(len(data_month)-12, 12, 1)
    dec_mark = torch.Tensor(dec_mark).reshape(len(data_month)-12, 12, 1)

    return enc_data, dec_data, target_data,enc_mark,dec_mark

train_data_enc, train_data_dec, train_data_out,enc_mark,dec_mark=data_gerenete(data_month_train_normelized)
train_data_enc=train_data_enc.to(device)
train_data_dec=train_data_dec.to(device)
train_data_out=train_data_out.to(device)
enc_mark=enc_mark.to(device)
dec_mark=dec_mark.to(device)

# train_data_enc=torch.randn([108,12,27]).to(device)
# train_data_dec=torch.randn([108,12,1]).to(device)
# train_data_out=torch.randn([108,1,1]).to(device)
#month embedding
# enc_mark=torch.randn([108,12,1]).to(device)
# dec_mark=torch.randn([108,12,1]).to(device)
criterion = nn.MSELoss()
lr = 0.0001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
train_epochs_loss=[]
for epoch in range(1000):
    print('epoch:',epoch)
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    train_epoch_loss=[]
    for i in range(0, len(train_data_enc)):
        targets=train_data_out[i].unsqueeze(0).to(device)
        enc=train_data_enc[i,:,:].unsqueeze(0).to(device)
        dec = train_data_dec[i,:,:].unsqueeze(0).to(device)
        enc=torch.concat([dec,enc_mark[i,:,:].unsqueeze(0)],dim=2).to(device)
        output = model(enc, enc_mark[i,:,:].unsqueeze(0), dec, dec_mark[i,:,:].unsqueeze(0))[0].to(device)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        train_epoch_loss.append(loss.item())
    train_epochs_loss.append(np.average(train_epoch_loss))
import matplotlib.pyplot as plt
plt.plot(range(epoch+1),train_epochs_loss)
plt.title('Autoformer')
plt.savefig('Autoformer_train')
# torch.save(model, "Transformer_model.pt")
# model(enc, enc_mark[i,:,:].unsqueeze(0), dec, dec_mark[i,:,:].unsqueeze(0))
#处理test数据
test_data_enc, test_data_dec, test_data_out,enc_mark,dec_mark=data_gerenete(data_month_test_normelized)
enc_mark=enc_mark.to(device)
dec_mark=dec_mark.to(device)
test_epoch_loss=[]
test_predict_squence=[]
test_truth_squence=[]
for i in range(len(test_data_out)):
    targets = test_data_out[i].unsqueeze(0).to(device)
    enc = test_data_enc[i, :, :].unsqueeze(0).to(device)
    dec = test_data_dec[i, :, :].unsqueeze(0).to(device)
    enc=torch.concat([dec,enc_mark[i,:,:].unsqueeze(0)],dim=2).to(device)
    # dec = torch.concat([dec, enc_mark[i, :, :].unsqueeze(0)], dim=2).to(device)
    output = model(enc, enc_mark[i, :, :].unsqueeze(0), dec, dec_mark[i, :, :].unsqueeze(0))[0].to(device)
    loss = criterion(output, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    optimizer.step()
    test_epoch_loss.append(loss.item())
    test_predict_squence.append(output)
    test_truth_squence.append(targets)

inv_predict_squence=[inverse_normalized(test_predict_squence[k],train_mean,train_std) for k in range(len(test_predict_squence))]
inv_truth_squence=[inverse_normalized(test_truth_squence[k],train_mean,train_std) for k in range(len(test_truth_squence))]

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
inv_predict_squence=[k.cpu().detach().numpy()[0][0][0] for k in inv_predict_squence]
inv_truth_squence=[k.cpu().detach().numpy()[0][0][0] for k in inv_truth_squence]
mse(inv_truth_squence, inv_predict_squence)
mape(inv_truth_squence, inv_predict_squence)
mae(inv_truth_squence, inv_predict_squence)
import pandas as pd
try:
    truth_predict_data=pd.read_excel('.\\truth_predict_data_S1.xlsx')
except:
    print('没找到文件')
    truth_predict_data=pd.DataFrame()
truth_predict_data['S1_Autoformer_t2v_truth']=inv_truth_squence
truth_predict_data['S1_Autoformer_t2v_predict']=inv_predict_squence
truth_predict_data.drop(columns='Unnamed: 0')
truth_predict_data.to_excel('truth_predict_data_S1.xlsx')
# import json
# S1_result=[mape(inv_truth_squence, inv_predict_squence),
#            mse(inv_truth_squence, inv_predict_squence),
#            mae(inv_truth_squence, inv_predict_squence)]
# with open('result_dict.json', 'r') as json_file:
# 	    result_dict = json.load(json_file)
# result_dict['S2_Autoformer_mape']=float(S1_result[0])
# result_dict['S2_Autoformer_mse']=float(S1_result[1])
# result_dict['S2_Autoformer_mae']=float(S1_result[2])
# #mape,mse,mae
# dict_json=json.dumps(result_dict)
# with open('result_dict.json','w+') as file:
#     file.write(dict_json)
# import json
# try:
#     with open('E:\用电量预测\models\\result_dict_no_t2v.json', 'r') as json_file:
# 	    result_dict = json.load(json_file)
# except:
#     print('没找到')
#     result_dict={}
# S1_result=[mape(inv_truth_squence, inv_predict_squence),
#            mse(inv_truth_squence, inv_predict_squence),
#            mae(inv_truth_squence, inv_predict_squence)]
# result_dict['S1_Autoformer_mape']=float(S1_result[0])
# result_dict['S1_Autoformer_mse']=float(S1_result[1])
# result_dict['S1_Autoformer_mae']=float(S1_result[2])
# #mape,mse,mae
# dict_json=json.dumps(result_dict)
# with open('result_dict_no_t2v.json','w+') as file:
#     file.write(dict_json)
