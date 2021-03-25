import torch
from torch import nn, optim
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, input_size, hid_size, output_size, num_hid_layers, init='box'):
        super().__init__()
        self.input_size = input_size
        self.hid_size = hid_size
        self.output_size = output_size
        self.num_hid_layers = num_hid_layers
        layers = []
        layers.append(nn.Linear(input_size, hid_size))
        for i in range(1, num_hid_layers):
            layers.append(nn.Linear(hid_size, hid_size))
        layers.append(nn.Linear(hid_size, output_size))
        self.layers = nn.ModuleList(layers)

        if init == 'box':
            self.init_box()
        elif init == 'debug':
            self.init_debug()

    def forward(self, x):
        for i in range(self.num_hid_layers):
            x= F.relu(self.layers[i].forward(x))
        return self.layers[-1].forward(x)

    def init_box(self):
        # box init
        for i in range(self.num_hid_layers):
            if i==0:
                A = torch.zeros((self.hid_size, self.input_size))
                b = torch.zeros(self.hid_size)
                p = torch.rand((self.hid_size, self.input_size))
                n = torch.normal(mean=torch.zeros((self.hid_size, self.input_size)), std=1)

            else:
                A = torch.zeros((self.hid_size, self.hid_size))
                b = torch.zeros(self.hid_size)
                p = torch.rand((self.hid_size, self.hid_size))
                n = torch.normal(mean=torch.zeros((self.hid_size, self.hid_size)), std=1)
            for j in range(self.hid_size):
                n[j,:] = n[j,:] / torch.norm(n[j,:], p=2)
                p_max = torch.maximum(torch.zeros(1), torch.sign(n[j,:]))
                k = 1/(torch.dot((p_max - p[j,:]), n[j,:]))
                A[j,:] = k * n[j,:]
                b[j] = -k * torch.dot(n[j,:], p[j,:])
            self.layers[i].weight.data = A
            self.layers[i].bias.data = b

        # Init last layer as identity map
        self.layers[-1].weight.data = torch.diag(torch.ones(self.hid_size))
        self.layers[-1].bias.data = torch.zeros(self.hid_size)


    def init_debug(self):
        # 3,3,1,1
        print("Debug initialization")
        self.layers[0].weight.data = torch.Tensor([[1,1,1],[0,1,1],[0,0,1]])
        self.layers[0].bias.data = torch.Tensor([0,0,0])
        self.layers[1].weight.data = torch.Tensor([[1,0,0]])
        self.layers[1].bias.data = torch.Tensor([0])
