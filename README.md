# Total-elsctricity-consumption-forecasting
Total electricity consumption forecasting based on Transformer  time series models

#Requirements
Dependencies can be installed using the following command:
pip install -r requirements.txt

#Data
The monthly dataset used in the paper were collected from Wind database.
TEC(Total electricity consumption) from January 2009 to December 2020.

#Methods
Three general Machine learning methods were used to serve as baseline models, including MLP, SVR, and XGboost.
Three variant models of the Transformer class are considered:Transformer, Informer, and Autoformer.
Time2Vec also was used to our model to embedding time feature.

#Result
By comparing with the existing Transformer models and other intelligent algorithm models, the robustness and superiority of the proposed method framework are verified, and the highest accuracy reaches 97.36%. The method presented in this paper provides valuable insights in the field of time series prediction。


#Citation
If you find this repository useful in your research, please consider citing the following paper and patent:

Paper:
Xuerong Li, Yiqiang Zhong, Wei Shang, Xun Zhang, Baoguo Shan, Xiang Wang,
Total electricity consumption forecasting based on Transformer time series models,
Procedia Computer Science,
Volume 214,
2022,
Pages 312-320,
ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2022.11.180.
(https://www.sciencedirect.com/science/article/pii/S1877050922018907)
Abstract: The total electricity consumption (TEC) reflects the operation condition of the national economy, and the prediction of the total electricity consumption can help track the economic development trend, as well as provide insights for macro policy making. Nowadays, novel neural networks provide a new perspective to predict the total electricity consumption. In this paper, a time series forecasting method based on Transformer model, Trans-T2V model, is proposed and applied to TEC forecasting. Transformer is the first network structure that completely relies on self-attention to calculate input and output. In this paper, the Time2vec method is used to improve the existing Transformer model, as embedding the month sequence more efficiently in the Transformer model. By comparing with the existing Transformer models and other intelligent algorithm models, the robustness and superiority of the proposed method framework are verified, and the highest accuracy reaches 97.36%. The method presented in this paper provides valuable insights in the field of time series prediction.
Keywords: Total electricity consumption; Transformer models; Forecasting framework; Time2vec embedding

Patent:
中国科学院数学与系统科学研究院等.基于季节-累积气温指数的全社会用电量预测方法和系统[P].中国专利:202211267347.3. 2022-10-17.
