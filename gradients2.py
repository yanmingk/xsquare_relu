import torch
import numpy as np
from class_dataset import MyDataSet
from class_model import MyModel
from train_model import train_with_LBFGS
from test_model import test_error, test, plot
from torch import nn
import matplotlib.pyplot as plt
import scipy.stats

import torch.onnx

from matplotlib.pyplot import semilogy
torch.set_default_dtype(torch.float64)

m=5


N_train = int(np.power(2,m+3))
X_train = torch.linspace(0,1,N_train).unsqueeze(1)
y_train = X_train**2

means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)
X_train_normalized = (X_train - means) / stds


criterion = nn.MSELoss()

train_losses = []
norm_g = []
model = MyModel(1, 3, 1, m, 'xavier')
train_with_LBFGS(model, criterion, X_train_normalized, y_train, 1, 0, train_losses, norm_g, record_g = 2, verbose = False)
(mean_err, max_err) = test_error(model, means, stds, npoints = int(np.power(2,m+2))+1, verbose = False)
maxlogerr = np.log2(max_err).numpy()

for i in range(m*2+2):

    # print(len(norm_g[i]))
    plt.subplot(2,m+1,1+i)
    norm_g_np = torch.cat(norm_g[i]).numpy()
    norm_g_np = norm_g_np/norm_g_np[0]
    minid = norm_g_np.size -1
    minval = norm_g_np[minid]
    print(f"{minid},{minval}")
    semilogy(list(range(len(norm_g_np))), norm_g_np, base=2)
    # plt.ylim((minval/10,10))
    plt.annotate(f"({str(minid)}, {str(minval)})",xytext=(0,0),xy=(minid,minval),textcoords='axes pixels')
    if(i<=m):
        plt.title(f"weight {i}")
    else:
        plt.title(f"bias {i-m-1}")

plt.annotate(f"({maxlogerr})",xytext=(0,0),xy=(minid,minval),textcoords='figure pixels')



for i in range(len(model.layers)):
    print(f"{model.layers[i].weight.grad.data}")
for i in range(len(model.layers)):
    print(f"{model.layers[i].bias.grad.data}")


plt.show()