import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

class LSGD():
    def __init__(self, model, losses, lr=1e-3):
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr)
        self.losses = losses

    def gd(self, trainset, batch_size, epochs = 1, criterion = nn.MSELoss() ):
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        for e in range(epochs):  
            self.model.train()
            for ind, data in enumerate(trainloader):
                inputs, labels = data['X'], data['y']

                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    if (ind==len(trainloader)-1): # record loss only after the last batch
                        self.losses.append(loss.detach().unsqueeze(0))
                    return loss
                self.optimizer.step(closure)
        return

    def ls(self, x, y, eps = 1e-16):
        # x: (n,1)
        # y: (n,1)
        self.model.train()
        states = self.model.get_states(x)
        # A: (n,m), m is number of basis functions
        # assume n > m, and A is full-rank
        A = torch.cat(states, 1)
        # A += eps
        # print(A.shape)
        sol = torch.transpose(torch.tensor(np.linalg.lstsq(A.numpy(),y.numpy(),None)[0]), 0, 1)
        self.model.layers[-1].weight.data = sol
        # xi_L = torch.lstsq(y, A)
        # self.model.layers[-1].weight.data = torch.transpose(xi_L.solution[:A.shape[1]], 0, 1)
