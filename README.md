# **Neural Chronos Ordinary Differential Equations, [arxiv preprint](https://arxiv.org/abs/2307.01023)**
## **C. Coelho, M. Fernanda P. Costa, L.L. Ferrás**

### **If you use this code, please cite our work:**

```
@article{coelho2023neural,
  title={Neural Chronos ODE: Unveiling Temporal Patterns and Forecasting Future and Past Trends in Time Series Data},
  author={Coelho, Cec{\i}lia and Costa, M Fernanda P and Ferr{\'a}s, Lu{\i}s L},
  journal={arXiv preprint arXiv:2307.01023},
  year={2023}
}
```

This work introduces Neural Chronos Ordinary Differential Equations (Neural CODE), a deep neural network architecture that fits a continuous-time ODE dynamics for predicting the chronology of a system both forward and backward in time. To train the model, we solve the ODE as an initial value problem and a final value problem, similar to Neural ODEs. We also explore two approaches to combining Neural CODE with Recurrent Neural Networks by replacing Neural ODE with Neural CODE (CODE-RNN), and incorporating a bidirectional RNN for full information flow in both time directions (CODE-BiRNN), and variants with other update cells namely GRU and LSTM: CODE-GRU, CODE-BiGRU, CODE-LSTM, CODE-BiLSTM.
Experimental results demonstrate that Neural CODE outperforms Neural ODE in learning the dynamics of a spiral forward and backward in time, even with sparser data. We also compare the performance of CODE-RNN/-GRU/-LSTM and CODE-BiRNN/-BiGRU/-BiLSTM against ODE-RNN/-GRU/-LSTM on three real-life time series data tasks: imputation of missing data for lower and higher dimensional data, and forward and backward extrapolation with shorter and longer time horizons. Our findings show that the proposed architectures converge faster, with CODE-BiRNN/-BiGRU/-BiLSTM consistently outperforming the other architectures on all tasks.

#### Examples Usage

In this repository, the architectures for the recurrent and latent architectures can be easily accessed by importing them from ```nnModels.py```. To use an architecture in this script, like CODE-RNN, you can import it as follows:

```python
import torch
from nnModels import CODE-RNN


```

The bidirectional spiral example using Neural CODE can be found in the ```ode_demo.py``` script. The script can be run with Neural CODE using the following command:

```
python ode_demo.py --nn arch 
```

where arch is the architecture name to be used. The available options are: NeuralCODE and NeuralODE.
Other options are available like batch_size, batch_time, number of iterations, etc. Check the script for more details.

The DJIA stock market example available for CODE-RNN, CODE-GRU, CODE-LSTM, CODE-BiGRU, CODE-BiRNN, and CODE-BiLSTM can be found in the ```timeSeriesDatasetRegression.py``` script. The script can be run using the following command:

```
python timeSeriesDatasetRegression.py arch_name DJIA flag
```

where arch_name is the architecture name to be used. The available options are: CODE-RNN, CODE-GRU, CODE-LSTM, CODE-BiGRU, CODE-BiRNN, and CODE-BiLSTM. The flag is a boolean value that indicates whether the data should be plotted and saved or not. If the flag is set to 1, the data will be plotted. Check the script for more details.

