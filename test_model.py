from class_dataset import MyDataSet
from class_model import MyModel

import numpy as np
import torch
import matplotlib.pyplot as plt


def test(model, X_test, y_test, criterion, verbose = False):

    model.eval()
    with torch.no_grad():
        output = model(X_test)
        loss = criterion(output, y_test)    
    if verbose:
        print(f"Test loss: { loss}")
    return loss


def test_error(model, X_mean, X_std, npoints = 100, verbose = False):
    if(npoints>100):
        X_ = torch.linspace(0, 1, npoints)
    else:
        X_ = torch.linspace(0, 1, 100)

    y_ = X_**2
    X_normalized = (X_.unsqueeze(1) - X_mean) / X_std
    y_output = model(X_normalized)

    errs = y_ - y_output.squeeze().data
    mean_err = torch.mean(abs(errs))
    max_err = torch.max(abs(errs))


    return mean_err, max_err


def plot(model, X_mean, X_std, pltpath,npoints = 100):
    X_ = torch.linspace(0, 1, npoints)
    y_ = X_**2
    X_normalized = (X_.unsqueeze(1) - X_mean) / X_std
    y_output = model(X_normalized)
    plt.cla()
    plt.plot(X_, y_)
    plt.plot(X_, y_output.squeeze().data)
    plt.savefig(pltpath)
