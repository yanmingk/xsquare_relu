import torch
import numpy as np
from class_dataset import MyDataSet
from class_model import MyModel
from train_model import train_with_SGD
from test_model import test_error, test, plot
from torch import nn
import matplotlib.pyplot as plt
import scipy.stats


criterion = nn.MSELoss()


mean_errors_m = []
max_errors_m = []
train_losses_m = []
norm_g_m = []
range_m = 8
num_models = 5


for m in range(7,range_m):

    N_train = int(np.power(2,m+3))
    X_train = torch.linspace(0,1,N_train).unsqueeze(1)
    y_train = X_train**2

    means = X_train.mean(dim=0, keepdim=True)
    stds = X_train.std(dim=0, keepdim=True)
    X_train_normalized = (X_train - means) / stds
    
    trainset = MyDataSet(X_train_normalized, y_train)
    
    mean_errors = []
    max_errors = []

    for j in range(num_models):
        train_losses = []
        norm_g = []
        model = MyModel(1, 3, 1, m, 'optimal')
        train_with_SGD(model, criterion, trainset,
                    int(np.max( [np.power(2.0,m), 2]) ), 8e-2, 0.9, 0.01, 20000, 0,
                    train_losses, norm_g, record_g = True, verbose=False)

        (mean_err, max_err) = test_error(model, means, stds, npoints = int(np.power(2,m+2))+1, verbose = False)
        train_losses_m.append(train_losses)
        norm_g_m.append(norm_g)

        mean_errors.append(mean_err)
        max_errors.append(max_err)
        print(f"\n{m} layers, run {j},"
                f"mean error: {np.log2(mean_err)}, "
                f"max error: {np.log2(max_err)}")
        pltpath = f"adam2_{m}-{j}"
        # plot(model, means, stds, pltpath, npoints = 100)

    max_errors_m.append(max_errors)
    mean_errors_m.append(mean_errors)


xaxis = np.array(list(range(7,range_m)))
y_theoretical = (-2)*xaxis-2

a = [np.log2(np.min(max_errors_m[i] ))for i in range(len(max_errors_m))]
b = [np.log2(scipy.stats.gmean(max_errors_m[i])) for i in range(len(max_errors_m))]

plt.cla()
plt.plot(xaxis, y_theoretical, 'b.-', label='theoretical')
plt.plot(xaxis, a , 'g.-', label='best trained result')
plt.plot(xaxis, b , 'm.-', label='average trained result')
plt.legend()
plt.xlabel('m')
plt.ylabel('${\log_2(\epsilon)}$')

plt.savefig(f'C:\\Users\\Yanming\\codes\\square\\SGD.png')



for m in range(7,range_m):
    for j in range(num_models):
        i = (m-1-6)*num_models + j

        print(len(train_losses_m[i]))
        plt.cla()
        plt.figure(figsize=(5,10))
        plt.subplot(4,1,1)
        plt.plot(list(range(len(train_losses_m[i]))), np.log10(torch.cat(train_losses_m[i]).numpy()))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(loss)}$')
        plt.subplot(4,1,2)
        plt.plot(list(range(len(train_losses_m[i]))), np.log10(torch.cat( [train_losses_m[i][j]-train_losses_m[i][-1] for j in range(len(train_losses_m[i]))] ).numpy()))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(loss-loss*)}$')
        plt.subplot(4,1,3)
        plt.plot(list(range(len(norm_g_m[i]))), np.log10(torch.cat(norm_g_m[i]).numpy()))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(grad)}$')
        plt.subplot(4,1,4)
        plt.plot(list(range(len(norm_g_m[i]))), np.log10(torch.cat([norm_g_m[i][j]-norm_g_m[i][-1] for j in range(len(norm_g_m[i]))]).numpy()))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(grad/grad[-1])}$')
        plt.savefig(f'figures/SGD20k-m{m}model{j}')

        n=50

        plt.cla()
        plt.figure(figsize=(5,10))
        plt.subplot(4,1,1)
        l = torch.cat( train_losses_m[i]).numpy()
        cumsum = np.cumsum(np.insert(l, 0, 0))
        moving_avg = (cumsum[n:] - cumsum[:-n]) / float(n)
        plt.plot(list(range(len(moving_avg))), np.log10(moving_avg))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(loss)}$_avg')
        plt.subplot(4,1,2)
        lastloss = train_losses_m[i][-1].numpy()
        # l = torch.cat( [train_losses_m[i][j]-train_losses_m[i][-1] for j in range(len(train_losses_m[i]))] ).numpy()
        # cumsum = np.cumsum(np.insert(l, 0, 0))
        # moving_avg = (cumsum[n:] - cumsum[:-n]) / float(n)
        plt.plot(list(range(len(moving_avg))), np.log10( [moving_avg[k]-moving_avg[-1] for k in range(len(moving_avg))]))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(loss-loss*)}$_avg')
        plt.subplot(4,1,3)
        l = torch.cat(norm_g_m[i]).numpy()
        cumsum = np.cumsum(np.insert(l, 0, 0))
        moving_avg = (cumsum[n:] - cumsum[:-n]) / float(n)
        plt.plot(list(range(len(moving_avg))), np.log10(moving_avg))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(grad)}$_avg')
        plt.subplot(4,1,4)
        lastgrad = norm_g_m[i][-1]
        # l = torch.cat([norm_g_m[i][j]-norm_g_m[i][-1] for j in range(len(norm_g_m[i]))]).numpy()
        # cumsum = np.cumsum(np.insert(l, 0, 0))
        # moving_avg = (cumsum[n:] - cumsum[:-n]) / float(n)
        plt.plot(list(range(len(moving_avg))), np.log10([moving_avg[k]-moving_avg[-1] for k in range(len(moving_avg))]))
        plt.xlabel('iteration')
        plt.ylabel('${\log_{10}(grad/grad[-1])}$_avg')
        plt.savefig(f'figures/SGD20k-m{m}model{j}_avg50')