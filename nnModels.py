import argparse
import numpy as np


import torch
import torch.nn as nn

#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint


# ODE FUNC implementation --------------------------------
class LatentODEfunc(nn.Module):

    def __init__(self, obs_dim, nhidden, latent_dim=0):
        super(LatentODEfunc, self).__init__()
        if latent_dim == 0: latent_dim = nhidden 
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, obs_dim+nhidden)
        self.fc2 = nn.Linear(obs_dim+nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out

class RevLatentODEfunc(nn.Module):

    def __init__(self, obs_dim, nhidden, latent_dim=0):
        super(RevLatentODEfunc, self).__init__()
        if latent_dim == 0: latent_dim = nhidden 
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, obs_dim+nhidden)
        self.fc2 = nn.Linear(obs_dim+nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return -out


# CODE-RNN implementation --------------------------------
class CODERNN(nn.Module):

    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(CODERNN, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim+ nhidden*2, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

    def forward(self, x, h_f, h_b, t_f, t_b):
        h_f = odeint(self.func, h_f, t_f)[-1]
        h_b = odeint(self.func, h_b, t_b)[-1]
        h = torch.cat((h_f, h_b))

        x = x.squeeze(0)

        combined = torch.cat((x, h))
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)  
        return out, h_f, h_b

    def initHidden(self):
        return torch.zeros(self.nhidden)



# CODE-GRU implementation --------------------------------
class CODEGRU(nn.Module):
    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(CODEGRU, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden*2, nhidden*2)
        self.h2o = nn.Linear(nhidden*2, out_dim)
        

    def forward(self, x, h_f, h_b, t_f, t_b):
        h_f = odeint(self.func, h_f, t_f)[-1]
        h_b = odeint(self.func, h_b, t_b)[-1]
        h = torch.cat((h_f, h_b))

        x = x.squeeze(0)
        combined = torch.cat((x, h))
        z = torch.sigmoid(self.i2h(combined))
        r = torch.sigmoid(self.i2h(combined))
        combined2 = torch.cat((x, r*h))
        h_hat = torch.tanh(self.i2h(combined2))
        h = z * h + (1 - z) * h_hat

        out = self.h2o(h)
        
        return out, h_f, h_b

    def initHidden(self):
        return torch.zeros(self.nhidden)


# CODE-LSTM implementation --------------------------------
class CODELSTM(nn.Module):
    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(CODELSTM, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden*2, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

    def forward(self, x, h_f, h_b, c, t_f, t_b):
        h_f = odeint(self.func, h_f, t_f)[-1]
        h_b = odeint(self.func, h_b, t_b)[-1]
        h = torch.cat((h_f, h_b))


        x = x.squeeze(0)
        combined = torch.cat((x, h))
        
        c_tilde = torch.sigmoid(self.i2h(combined))
        i = torch.sigmoid(self.i2h(combined))
        f = torch.sigmoid(self.i2h(combined))
        o = torch.sigmoid(self.i2h(combined))

        c = f * c + i * c_tilde 

        h = o * torch.tanh(c)
        out = self.h2o(h)
        
        return out, h_f, h_b, c

    def initHidden(self):
        return torch.zeros(self.nhidden)


# CODE-BiRNN implementation --------------------------------
class CODEBiRNN(nn.Module):

    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(CODEBiRNN, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim+ nhidden, nhidden)
        self.h2o = nn.Linear(nhidden*2, out_dim)

    def forward(self, x_f, x_b, h_f, h_b, t_f, t_b):
        h_f = odeint(self.func, h_f, t_f)[-1]
        h_b = odeint(self.func, h_b, t_b)[-1]

        x_f = x_f.squeeze(0)
        combined_f = torch.cat((x_f, h_f))
        h_f = torch.tanh(self.i2h(combined_f))

        x_b = x_b.squeeze(0)
        combined_b = torch.cat((x_b, h_b))
        h_b = torch.tanh(self.i2h(combined_b))

        h = torch.cat((h_f, h_b))
        out = self.h2o(h)  
        return out, h_f, h_b

    def initHidden(self):
        return torch.zeros(self.nhidden)


# CODE-BiGRU implementation --------------------------------
class CODEBiGRU(nn.Module):
    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(CODEBiGRU, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden*2, out_dim)
        

    def forward(self, x_f, x_b, h_f, h_b, t_f, t_b):
        h_f = odeint(self.func, h_f, t_f)[-1]
        h_b = odeint(self.func, h_b, t_b)[-1]

        x_f = x_f.squeeze(0)
        combined_f = torch.cat((x_f, h_f))
        z_f = torch.sigmoid(self.i2h(combined_f))
        r_f = torch.sigmoid(self.i2h(combined_f))
        combined2_f = torch.cat((x_f, r_f*h_f))
        h_hat_f = torch.tanh(self.i2h(combined2_f))
        h_f = z_f * h_f + (1 - z_f) * h_hat_f

        x_b = x_b.squeeze(0)
        combined_b = torch.cat((x_b, h_b))
        z_b = torch.sigmoid(self.i2h(combined_b))
        r_b = torch.sigmoid(self.i2h(combined_b))
        combined2_b = torch.cat((x_b, r_b*h_b))
        h_hat_b = torch.tanh(self.i2h(combined2_b))
        h_b = z_b * h_b + (1 - z_b) * h_hat_b

        h = torch.cat((h_f, h_b))

        out = self.h2o(h)
        
        return out, h_f, h_b 

    def initHidden(self):
        return torch.zeros(self.nhidden)


# CODE-LSTM implementation --------------------------------
class CODEBiLSTM(nn.Module):
    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(CODEBiLSTM, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden*2, out_dim)

        

    def forward(self, x_b, x_f, h_f, h_b, c_f, c_b, t_f, t_b):
        h_f = odeint(self.func, h_f, t_f)[-1]
        h_b = odeint(self.func, h_b, t_b)[-1]


        x_f = x_f.squeeze(0)
        combined_f = torch.cat((x_f, h_f))
        c_tilde_f = torch.sigmoid(self.i2h(combined_f))
        i_f = torch.sigmoid(self.i2h(combined_f))
        f_f = torch.sigmoid(self.i2h(combined_f))
        o_f = torch.sigmoid(self.i2h(combined_f))
        c_f = f_f * c_f + i_f * c_tilde_f 
        h_f = o_f * torch.tanh(c_f)

        x_b = x_b.squeeze(0)
        combined_b = torch.cat((x_b, h_b))
        c_tilde_b = torch.sigmoid(self.i2h(combined_b))
        i_b = torch.sigmoid(self.i2h(combined_b))
        f_b = torch.sigmoid(self.i2h(combined_b))
        o_b = torch.sigmoid(self.i2h(combined_b))
        c_b = f_b * c_b + i_b * c_tilde_b 
        h_b = o_b * torch.tanh(c_b)

        h = torch.cat((h_f, h_b))
        out = self.h2o(h)
        
        return out, h_f, h_b, c_f, c_b

    def initHidden(self):
        return torch.zeros(self.nhidden)





