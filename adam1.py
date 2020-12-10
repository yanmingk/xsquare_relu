import torch
import numpy as np
from class_dataset import MyDataSet
from class_model import MyModel
from train_model import train_with_adam
from test_model import test_error, test, plot
from torch import nn
import matplotlib.pyplot as plt
import scipy.stats


criterion = nn.MSELoss()


mean_errors_m = []
max_errors_m = []
train_losses_m = []
norm_g_m = []

for m in range(1,8):

    N_train = int(np.power(2,m+3))
    X_train = torch.linspace(0,1,N_train).unsqueeze(1)
    y_train = X_train**2

    means = X_train.mean(dim=0, keepdim=True)
    stds = X_train.std(dim=0, keepdim=True)
    X_train_normalized = (X_train - means) / stds
    
    trainset = MyDataSet(X_train_normalized, y_train)
    
    mean_errors = []
    max_errors = []

    for j in range(10):
        train_losses = []
        norm_g = []
        model = MyModel(1, 3, 1, m, 'optimal')
        train_with_adam(model, criterion, trainset,
                    int(np.max( [np.power(2.0,m), 2]) ), 1e-2, 5000, 0,#np.power(10.0,-2*m-2),#-1 -3 int(np.power(10.0,-m-2)),#1e-3, 
                    train_losses, norm_g, record_g = False , verbose=False)

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


xaxis = np.array(list(range(1,8)))
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

plt.savefig(f'C:\\Users\\Yanming\\codes\\square\\adamoptimalnormalized1e-2.png')