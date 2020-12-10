import torch
import numpy as np
from class_dataset import MyDataSet
from class_model import MyModel
from train_model import train_with_LBFGS, train_with_SGD
from test_model import test_error, test, plot
from torch import nn
import matplotlib.pyplot as plt
import scipy.stats

torch.set_default_dtype(torch.float64)

criterion = nn.MSELoss()
range_m = 5
num_models = 1

mean_errors_m = []
max_errors_m = []
train_losses_m = []
norm_g_m = []


for m in range(1,range_m):

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
        model = MyModel(1, 3, 1, m, 'xavier')
        last_loss = 10000
        n = 500
        for i in range(n):
            current_loss = train_with_LBFGS(model, criterion, X_train_normalized, y_train, 1, 0, train_losses, norm_g, record_g = 0, verbose = False)

            print(f"Iteration {i} --- LBFGS loss: {current_loss.data.numpy()}")

            # go back to last model after L-BFGS
            if (last_loss < current_loss):
                print(f"recover last model")
                checkpoint = torch.load( "bfgs_sgd.pt")
                model.load_state_dict(checkpoint['model_state_dict'])
                current_loss = checkpoint['loss']
            # loss improves
            else:
                last_model = model
                torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'loss': current_loss,
                }, "bfgs_sgd.pt")

            # improved <= 0.1%
            if( (last_loss - current_loss) < 1e-3 * last_loss) and (i < n-1):
                print(f"train with SGD")  
                batch_size = int(np.max( [np.power(2.0,m-1), 1] ))
                lr = 1e-2
                epochs = 100
                train_with_SGD(model, criterion, trainset, batch_size, lr, 0.9, 0, epochs, 0, train_losses , norm_g , record_g = False)

            if (last_loss > current_loss):
                last_loss = current_loss




        (mean_err, max_err) = test_error(model, means, stds, npoints = int(np.power(2,m+2))+1, verbose = False)
        train_losses_m.append(train_losses)

        mean_errors.append(mean_err)
        max_errors.append(max_err)
        print(f"\n{m} layers, run {j},"
                f"mean error: {np.log2(mean_err)}, "
                f"max error: {np.log2(max_err)}")


    max_errors_m.append(max_errors)
    mean_errors_m.append(mean_errors)


for m in range(1,range_m):
    for j in range(num_models):
        i = (m-1)*num_models + j
        print(len(train_losses_m[i]))
        plt.cla()
        plt.plot(list(range(len(train_losses_m[i]))), np.log10(torch.cat(train_losses_m[i]).numpy()))
        plt.xlabel('iteration')
        plt.ylabel('${\log_10(loss)}$')
        plt.savefig(f'C:\\Users\\Yanming\\codes\\square\\losses\\bfgs-sgd{m}model{j}')






xaxis = np.array(list(range(1,5)))
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

plt.savefig(f'C:\\Users\\Yanming\\codes\\square\\bfg-sgd.png')
