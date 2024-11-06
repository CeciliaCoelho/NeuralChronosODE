# **Neural Chronos Ordinary Differential Equations, [arxiv preprint](https://arxiv.org/abs/2307.01023)**
## **C. Coelho, M. Fernanda P. Costa, L.L. Ferrás**

This work introduces Neural Chronos Ordinary Differential Equations (Neural CODE), a deep neural network architecture that fits a continuous-time ODE dynamics for predicting the chronology of a system both forward and backward in time. To train the model, we solve the ODE as an initial value problem and a final value problem, similar to Neural ODEs. We also explore two approaches to combining Neural CODE with Recurrent Neural Networks by replacing Neural ODE with Neural CODE (CODE-RNN), and incorporating a bidirectional RNN for full information flow in both time directions (CODE-BiRNN), and variants with other update cells namely GRU and LSTM: CODE-GRU, CODE-BiGRU, CODE-LSTM, CODE-BiLSTM.
Experimental results demonstrate that Neural CODE outperforms Neural ODE in learning the dynamics of a spiral forward and backward in time, even with sparser data. We also compare the performance of CODE-RNN/-GRU/-LSTM and CODE-BiRNN/-BiGRU/-BiLSTM against ODE-RNN/-GRU/-LSTM on three real-life time series data tasks: imputation of missing data for lower and higher dimensional data, and forward and backward extrapolation with shorter and longer time horizons. Our findings show that the proposed architectures converge faster, with CODE-BiRNN/-BiGRU/-BiLSTM consistently outperforming the other architectures on all tasks.


