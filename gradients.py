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
train_with_LBFGS(model, criterion, X_train, y_train, 1, 0, train_losses, norm_g, record_g = 1, verbose = False)

(mean_err, max_err) = test_error(model, 0, 1, npoints = int(np.power(2,m+2))+1, verbose = False)
maxlogerr = np.log2(max_err).numpy()

# print(norm_g)
norm_g_np = torch.cat(norm_g).numpy()
norm_g_np = norm_g_np/norm_g_np[0]
minid = norm_g_np.size-1
minval = norm_g_np[minid]
print(f"{minid},{minval}")
semilogy(list(range(len(norm_g_np))), norm_g_np, base=2)
# semilogy(list(range(len(norm_g))),[norm_g[i]/norm_g[0] for i in range(len(norm_g))])
plt.ylim((minval/10,10))
plt.annotate(f"({str(minid)}, {str(minval)}):{maxlogerr}",xytext=(0,0),xy=(minid,minval),textcoords='axes pixels')
plt.ylabel("norm(g[i])/norm(g[0])")
plt.xlabel("iteration")




plt.show()