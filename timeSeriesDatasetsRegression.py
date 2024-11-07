import pandas as pd
from datetime import datetime
import calendar
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from nnModels import *

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dateConvert(initDate, date):

    conv = (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(initDate, "%Y-%m-%d")).days

    return conv



def plotData(x, pred_t, pred_x, path):

    figure, axis = plt.subplots(len(pred_x[0]), 1)

    for f in range(len(pred_x[0])):

        true = [j[f].item() for i in x for j in i]
        predicted = [i[f].item() for i in pred_x]
        


        axis[f].plot(pred_t, true, color='b')
        axis[f].plot(pred_t, predicted, color='r')

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    plt.savefig(path)

def splitData(t, x):

    try:
        x = x.astype(float).to_list()
    except:
        x = [list(map(lambda y: y.astype(float).tolist(),i)) for i in x]

    t = t.to_list()

    ind = [[i,i+1,i+2] for i in range(0,len(t)-3, 3)]

    x_data = [torch.Tensor([x[i[0]],x[i[1]],x[i[2]]]) for i in ind]
    t_data = [torch.Tensor([t[i[0]],t[i[1]],t[i[2]]]) for i in ind]

    ind_t = random.sample(range(len(t_data)), int(len(t_data)*0.8))

    x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2, random_state=36)

    return x_train, x_test, t_train, t_test



def train(x_train, t_train, arch):

    model.zero_grad()
    optimizer.zero_grad()

    x0 = x_train[0]
    x0_b = x_train[-1]
    x = torch.Tensor([x0]).to(device)


    if arch == 'CODE-BiRNN' or arch == 'CODE-BiGRU':
            h_f = model.initHidden().to(device)
            h_b = model.initHidden().to(device)
            for i in range(1, len(x_train)):
                t = torch.Tensor([t_train[i-1], t_train[i]])
                x_b = x_train[i]
                output, h_f, h_b = model(x0.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                x0 = x_train[i]
                x = torch.cat((x, output), 0) 
            output = x
    elif arch == 'CODE-BiLSTM':
            h_f = model.initHidden().to(device)
            h_b = model.initHidden().to(device)
            c_f = model.initHidden().to(device)
            c_b = model.initHidden().to(device)
            for i in range(1, len(x_train)):
                t = torch.Tensor([t_train[i-1], t_train[i]])
                x_b = x_train[i]
                output, h_f, h_b, c_f, c_b = model(x0.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, c_f, c_b, t, reversed(t))
                x0 = x_train[i]
                x = torch.cat((x, output), 0) 
            output = x
    elif arch == 'CODE-RNN' or arch == 'CODE-GRU':
            h_f = model.initHidden().to(device)
            h_b = model.initHidden().to(device)
            for i in range(1, len(x_train)):
                t = torch.Tensor([t_train[i-1], t_train[i]])
                output, h_f, h_b = model(x0.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                x0 = x_train[i]
                x = torch.cat((x, output), 0) 
            output = x
    elif arch == 'CODE-LSTM':
            h_f = model.initHidden().to(device)
            h_b = model.initHidden().to(device)
            c = model.initHidden().to(device)
            for i in range(1, len(x_train)):
                t = torch.Tensor([t_train[i-1], t_train[i]])
                output, h_f, h_b, c = model(x0.unsqueeze(0).to(device), h_f, h_b, c, t, reversed(t))
                x0 = x_train[i]
                x = torch.cat((x, output), 0) 
            output = x

    loss = criterion(output.to(device), x_train.squeeze(1).to(device))
    loss.backward()
    optimizer.step()


    return output, loss.item()


if __name__ == '__main__':

    arch = sys.argv[1]
    data = sys.argv[2]
    savePlots = sys.argv[3]
    noise_std = .3

    if data == 'DJIA':
        path = 'AAPL_2006-01-01_to_2018-01-01.csv'
        df = pd.read_csv(path)
        df['Date'] = [dateConvert("2006-01-01", f) for f in df['Date']]
        df = df.drop('Name', axis=1)
        df = df.apply(lambda x: x/x.max())
        x_train, x_test, t_train, t_test = splitData(df['Date'], df[['Close']].values)
    iters = 50 
    train_split = 0.75

    if arch == 'CODE-BiRNN':
        func = LatentODEfunc(1, 256).to(device)
        model = CODEBiRNN(func, 1, 256, 1).to(device)
    elif arch == 'CODE-BiGRU':
        func = LatentODEfunc(1, 256).to(device)
        model = CODEBiGRU(func, 1, 256, 1).to(device)
    elif arch == 'CODE-BiLSTM':
        func = LatentODEfunc(1, 256).to(device)
        model = CODEBiLSTM(func, 1, 256, 1).to(device)
    elif arch == 'CODE-RNN':
        func = LatentODEfunc(1, 256).to(device)
        model = CODERNN(func, 1, 256, 1).to(device)
    elif arch == 'CODE-GRU':
        func = LatentODEfunc(1, 256).to(device)
        model = CODEGRU(func, 1, 256, 1).to(device)
    elif arch == 'CODE-LSTM':
        func = LatentODEfunc(1, 256).to(device)
        model = CODELSTM(func, 1, 256, 1).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 


    print("Starting to train with a ", arch)
    train_stime = time.time()

    for n in range(iters):
        print("**ITER ", n, "**")
        for i in range(0, int(len(x_train))):
            output, loss = train(x_train[i], t_train[i], arch)

            if i % 1000 == 0:
                print("Loss: ", loss, "Predicted: ", output, "Expected: ", x_train[i].squeeze(1))

                
                with torch.no_grad():

                    tMse = 0
                    tMseb = 0
                    pred_x = []
                    pred_t = []

                    for j in range(len(x_train)):
                        outputT = [x_train[j][0]]
                        outputB = [x_train[j][-1]]

                        if arch == 'CODE-BiRNN' or arch == "CODE-BiGRU":
                                h_f = model.initHidden().to(device)
                                h_b = model.initHidden().to(device)
                                x = x_train[j][0]
                                for i in range(1, len(x_train[j])):
                                    t = torch.Tensor([t_train[j][i-1], t_train[j][i]])
                                    x_b = x_train[j][i]
                                    output, h_f, h_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                                    x = output
                                    outputT.append(output)
                                outputt = torch.tensor(outputT, requires_grad=True)
                                x = x_train[j][-1]
                                for i in range(len(x_train[j])-1, 0, -1):
                                    t = torch.Tensor([t_train[j][i], t_train[j][i-1]])
                                    x_b = x_train[j][i]
                                    output, h_f, h_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                                    x = output
                                    outputB.append(output)
                                outputb = torch.tensor(outputB, requires_grad=True)
                        elif arch == 'CODE-BiLSTM':
                                h_f = model.initHidden().to(device)
                                h_b = model.initHidden().to(device)
                                c_f = model.initHidden().to(device)
                                c_b = model.initHidden().to(device)
                                x = x_train[j][-1]
                                for i in range(1, len(x_train[j])):
                                    t = torch.Tensor([t_train[j][i-1], t_train[j][i]])
                                    x_b = x_train[j][i]
                                    output, h_f, h_b, c_f, c_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, c_f, c_b, t, reversed(t))
                                    x = output
                                    outputT.append(output)
                                outputt = torch.tensor(outputT, requires_grad=True)
                                x = x_train[j][0]
                                for i in range(len(x_train[j])-1, 0, -1):
                                    t = torch.Tensor([t_train[j][i], t_train[j][i-1]])
                                    x_b = x_train[j][i]
                                    output, h_f, h_b, c_f, c_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, c_f, c_b, t, reversed(t))
                                    x = output
                                    outputB.append(output)
                                outputb = torch.tensor(outputB, requires_grad=True)
                        elif arch == 'CODE-RNN' or arch == "CODE-GRU":
                                h_f = model.initHidden().to(device)
                                h_b = model.initHidden().to(device)
                                x = x_train[j][0]
                                for i in range(1, len(x_train[j])):
                                    t = torch.Tensor([t_train[j][i-1], t_train[j][i]])
                                    output, h_f, h_b = model(x.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                                    x = output
                                    outputT.append(output)
                                outputt = torch.tensor(outputT, requires_grad=True)
                                x = x_train[j][-1]
                                for i in range(len(x_train[j])-1, 0, -1):
                                    t = torch.Tensor([t_train[j][i], t_train[j][i-1]])
                                    output, h_f, h_b = model(x.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                                    x = output
                                    outputB.append(output)
                                outputb = torch.tensor(outputB, requires_grad=True)
                        elif arch == 'CODE-LSTM':
                                h_f = model.initHidden().to(device)
                                h_b = model.initHidden().to(device)
                                c = model.initHidden().to(device)
                                x = x_train[j][0]
                                for i in range(1, len(x_train[j])):
                                    t = torch.Tensor([t_train[j][i-1], t_train[j][i]])
                                    output, h_f, h_b, c = model(x.unsqueeze(0).to(device), h_f, h_b, c, t, reversed(t))
                                    x = output
                                    outputT.append(output)
                                outputt = torch.tensor(outputT, requires_grad=True)
                                x = x_train[j][-1]
                                for i in range(len(x_train[j])-1, 0, -1):
                                    t = torch.Tensor([t_train[j][i], t_train[j][i-1]])
                                    output, h_f, h_b, c = model(x.unsqueeze(0).to(device), h_f, h_b, c, t, reversed(t))
                                    x = output
                                    outputB.append(output)
                                outputb = torch.tensor(outputB, requires_grad=True)

                        pred_x += outputt
                        pred_t += t_train[j]
                        mse = nn.MSELoss()(outputt.to(device), x_train[j].squeeze(1).to(device))
                        mseb = nn.MSELoss()(outputb.to(device), reversed(x_train[j].squeeze(1).to(device)))
                        tMse += mse
                        tMseb += mseb




                    print("back Loss: ", tMseb.item()/(len(x_test)-int(len(x_test)*train_split)))
                

    train_ftime = time.time()
    # Testing loop ---------------------------------------------

    with torch.no_grad():

        print("Testing starts now...")
        test_stime = time.time()

        tMse = 0
        tMseb = 0
        pred_x = []
        pred_t = []

        for j in range(len(x_test)):
            outputT = [x_test[j][0]]
            outputB = [x_test[j][-1]]

            if arch == 'CODE-BiRNN' or arch == "CODE-BiGRU":
                    h_f = model.initHidden().to(device)
                    h_b = model.initHidden().to(device)
                    x = x_test[j][0]
                    for i in range(1, len(x_test[j])):
                        t = torch.Tensor([t_test[j][i-1], t_test[j][i]])
                        x_b = x_test[j][i]
                        output, h_f, h_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                        x = output
                        outputT.append(output)
                    outputt = torch.tensor(outputT, requires_grad=True)
                    x = x_test[j][-1]
                    for i in range(len(x_test[j])-1, 0, -1):
                        t = torch.Tensor([t_test[j][i], t_test[j][i-1]])
                        x_b = x_test[j][i]
                        output, h_f, h_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                        x = output
                        outputB.append(output)
                    outputb = torch.tensor(outputB, requires_grad=True)
            elif arch == 'CODE-BiLSTM':
                    h_f = model.initHidden().to(device)
                    h_b = model.initHidden().to(device)
                    c_f = model.initHidden().to(device)
                    c_b = model.initHidden().to(device)
                    x = x_test[j][-1]
                    for i in range(1, len(x_test[j])):
                        t = torch.Tensor([t_test[j][i-1], t_test[j][i]])
                        x_b = x_test[j][i]
                        output, h_f, h_b, c_f, c_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, c_f, c_b, t, reversed(t))
                        x = output
                        outputT.append(output)
                    outputt = torch.tensor(outputT, requires_grad=True)
                    x = x_test[j][0]
                    for i in range(len(x_test[j])-1, 0, -1):
                        t = torch.Tensor([t_test[j][i], t_test[j][i-1]])
                        x_b = x_test[j][i]
                        output, h_f, h_b, c_f, c_b = model(x.unsqueeze(0).to(device), x_b.unsqueeze(0).to(device), h_f, h_b, c_f, c_b, t, reversed(t))
                        x = output
                        outputB.append(output)
                    outputb = torch.tensor(outputB, requires_grad=True)
            elif arch == 'CODE-RNN' or arch == "CODE-GRU":
                    h_f = model.initHidden().to(device)
                    h_b = model.initHidden().to(device)
                    x = x_test[j][0]
                    for i in range(1, len(x_test[j])):
                        t = torch.Tensor([t_test[j][i-1], t_test[j][i]])
                        output, h_f, h_b = model(x.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                        x = output
                        outputT.append(output)
                    outputt = torch.tensor(outputT, requires_grad=True)
                    x = x_test[j][-1]
                    for i in range(len(x_test[j])-1, 0, -1):
                        t = torch.Tensor([t_test[j][i], t_test[j][i-1]])
                        output, h_f, h_b = model(x.unsqueeze(0).to(device), h_f, h_b, t, reversed(t))
                        x = output
                        outputB.append(output)
                    outputb = torch.tensor(outputB, requires_grad=True)
            elif arch == 'CODE-LSTM':
                    h_f = model.initHidden().to(device)
                    h_b = model.initHidden().to(device)
                    c = model.initHidden().to(device)
                    x = x_test[j][0]
                    for i in range(1, len(x_test[j])):
                        t = torch.Tensor([t_test[j][i-1], t_test[j][i]])
                        output, h_f, h_b, c = model(x.unsqueeze(0).to(device), h_f, h_b, c, t, reversed(t))
                        x = output
                        outputT.append(output)
                    outputt = torch.tensor(outputT, requires_grad=True)
                    x = x_test[j][-1]
                    for i in range(len(x_test[j])-1, 0, -1):
                        t = torch.Tensor([t_test[j][i], t_test[j][i-1]])
                        output, h_f, h_b, c = model(x.unsqueeze(0).to(device), h_f, h_b, c, t, reversed(t))
                        x = output
                        outputB.append(output)
                    outputb = torch.tensor(outputB, requires_grad=True)

            pred_x += outputt
            pred_t += t_test[j]
            mse = nn.MSELoss()(outputt.to(device), x_test[j].squeeze(1).to(device))
            mseb = nn.MSELoss()(outputb.to(device), reversed(x_test[j].squeeze(1).to(device)))
            tMse += mse
            tMseb += mseb


            print("MSE: ", mse.item(), "Predicted: ", outputt, "Expected: ", x_test[j].squeeze(1))
            print("backwards MSE: ", mseb.item(), "Predicted: ", outputb, "Expected: ", reversed(x_test[j].squeeze(1)))


        print("Testing MSE: ", tMse.item()/(len(x_test)-int(len(x_test)*train_split)))
        print("Testing MSE backwards: ", tMseb.item()/(len(x_test)-int(len(x_test)*train_split)))
        test_ftime = time.time()
    
    print("elapsed time druing training: ", train_ftime-train_stime)
    print("elapsed time druing testing: ", test_ftime-test_stime)
    if savePlots == '1':
        path = "DJIA_dataset/" + arch + "_" + datetime.now().strftime('%s') + ".png" 
        plotData(x_test[int(len(x_test)*train_split):], pred_t, pred_x, path)
