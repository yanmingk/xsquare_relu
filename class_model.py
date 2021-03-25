import torch
from torch import nn, optim
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_size, hid_size, output_size, num_hid_layers, init='optimal', lsgd=False):
        super().__init__()

        self.input_size = input_size
        self.hid_size = hid_size
        self.output_size = output_size
        self.num_hid_layers = num_hid_layers
        self.lsgd = lsgd

        layers = []
        layers.append(nn.Linear(input_size, hid_size))
        for i in range(1, num_hid_layers):
            layers.append(nn.Linear(hid_size, hid_size))
        if lsgd:
            layers.append(nn.Linear(input_size + (num_hid_layers) * hid_size, output_size, bias = False))
        else:
            layers.append(nn.Linear(input_size + (num_hid_layers) * hid_size, output_size))


        self.layers = nn.ModuleList(layers)

        if(init == 'optimal'):
            self.init_optimal()
        elif(init == 'box'):
            self.init_box()
        elif(init == 'xavier_uniform'):
            self.init_xavier_uniform()
        elif(init == 'xavier_normal'):
            self.init_xavier_normal()
        elif init == 'kaiming_uniform':
            self.init_kaiming_uniform()
        elif(init == 'kaiming_normal'):
            self.init_kaiming_normal()

        # elif(init == 'normal'):
        #     self.init_normal()
        # elif(init == 'zeros'):
        #     self.init_zeros()
        # elif(init == 'ones'):
        #     self.init_ones()
        # elif(init == 'optimal_perturb'):
        #     self.init_optimal_perturb(1e-2)
        # elif(init == 'optimal_outputlayer'):
        #     self.init_xavier()
        #     self.init_optimal_outputlayer()
        
    def forward(self, x):
        states = [x]     
        for i in range(self.num_hid_layers):
            states.append( F.relu(self.layers[i].forward(states[i])))
        # input needs to have two dimensions
        all_states = torch.cat(states, 1)
        x = self.layers[-1].forward(all_states)
        return x

    def get_states(self, x):
        states = [x]     
        for i in range(self.num_hid_layers):
            states.append( F.relu(self.layers[i].forward(states[i])).detach())
        return states

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

        # Init last layer as all ones 
        self.layers[-1].weight.data = \
            torch.ones((self.output_size, self.input_size + (self.num_hid_layers) * self.hid_size))
        if self.lsgd:
            self.layers[-1].weight.requires_grad = False
        else:
            self.layers[-1].bias.data = torch.ones(self.output_size)


    def init_xavier_uniform(self):
        for i in range(len(self.layers)):
            torch.nn.init.xavier_uniform_(self.layers[i].weight)
            if i != len(self.layers) - 1 and self.lsgd == False:
                torch.nn.init.zeros_(self.layers[i].bias)

    def init_xavier_normal(self):
        for i in range(len(self.layers)):
            torch.nn.init.xavier_normal_(self.layers[i].weight)
            if i != len(self.layers) - 1 and self.lsgd == False:
                torch.nn.init.zeros_(self.layers[i].bias)

    def init_kaiming_uniform(self):
        for i in range(len(self.layers)):
            torch.nn.init.kaiming_uniform_(self.layers[i].weight, nonlinearity='relu')
            if i != len(self.layers) - 1 and self.lsgd == False:
                torch.nn.init.zeros_(self.layers[i].bias)

    def init_kaiming_normal(self):
        for i in range(len(self.layers)):
            torch.nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            if i != len(self.layers) - 1 and self.lsgd == False:
                torch.nn.init.zeros_(self.layers[i].bias)

    def init_optimal(self):
        # input_size = 1
        # hid_size = 3
        # output_size = 1

        with torch.no_grad():
            self.layers[0].weight.data = torch.Tensor([[1],[1],[1]])
            self.layers[0].bias.data = torch.Tensor([0,-0.5,-1])

            for i in range(1, len(self.layers) -1):
                self.layers[i].weight.data = torch.Tensor([[2,-4,2],[2,-4,2],[2,-4,2]])
                self.layers[i].bias.data = torch.Tensor([0,-0.5,-1])
        
            output_layer_weights = []
            output_layer_weights.append( torch.Tensor([[1]]))
            for i in range(self.num_hid_layers):
                output_layer_weights.append( torch.pow( torch.Tensor([[2]]), -2*(i+1) ) * torch.Tensor([[-2,4,-2]]))
            
            self.layers[-1].weight.data = torch.cat(output_layer_weights, 1)
            self.layers[-1].bias.data = torch.Tensor([0])    
        
    # def init_optimal_perturb(self, std):
    #     with torch.no_grad():
    #         self.layers[0].weight.data = torch.Tensor([[1],[1],[1]]) + std*torch.randn(3,1)
    #         self.layers[0].bias.data = torch.Tensor([0,-0.5,-1]) + std*torch.randn(3)

    #         for i in range(1, len(self.layers) -1):
    #             self.layers[i].weight.data = torch.Tensor([[2,-4,2],[2,-4,2],[2,-4,2]]) + std*torch.randn(3,3)
    #             self.layers[i].bias.data = torch.Tensor([0,-0.5,-1]) + std*torch.randn(3)
        
    #         output_layer_weights = []
    #         output_layer_weights.append( torch.Tensor([[1]]) + std*torch.randn(1,1))
    #         for i in range(self.num_hid_layers):
    #             output_layer_weights.append( torch.pow( torch.Tensor([[2]]), -2*(i+1) ) * torch.Tensor([[-2,4,-2]]) + std*torch.randn(1,3))
            
    #         self.layers[-1].weight.data = torch.cat(output_layer_weights, 1)
    #         self.layers[-1].bias.data = torch.Tensor([0])  + std*torch.randn(1)  


    # def init_zeros(self):
    #     with torch.no_grad():
    #         self.layers[0].weight.data = torch.Tensor([[0],[0],[0]])
    #         self.layers[0].bias.data = torch.Tensor([0,0,0])

    #         for i in range(1, len(self.layers) -1):
    #             self.layers[i].weight.data = torch.Tensor([[0,0,0],[0,0,0],[0,0,0]])
    #             self.layers[i].bias.data = torch.Tensor([0,0,0])
        
    #         output_layer_weights = []
    #         output_layer_weights.append( torch.Tensor([[0]]))
    #         for i in range(self.num_hid_layers):
    #             output_layer_weights.append( torch.Tensor([[0,0,0]]))
            
    #         self.layers[-1].weight.data = torch.cat(output_layer_weights, 1)
    #         self.layers[-1].bias.data = torch.Tensor([0])    
    # def init_ones(self):
    #     with torch.no_grad():
    #         self.layers[0].weight.data = torch.Tensor([[1],[1],[1]])
    #         self.layers[0].bias.data = torch.Tensor([0,0,0])

    #         for i in range(1, len(self.layers) -1):
    #             self.layers[i].weight.data = torch.Tensor([[1,1,1],[1,1,1],[1,1,1]])
    #             self.layers[i].bias.data = torch.Tensor([0,0,0])
        
    #         output_layer_weights = []
    #         output_layer_weights.append( torch.Tensor([[1]]))
    #         for i in range(self.num_hid_layers):
    #             output_layer_weights.append( torch.Tensor([[1,1,1]]))
            
    #         self.layers[-1].weight.data = torch.cat(output_layer_weights, 1)
    #         self.layers[-1].bias.data = torch.Tensor([0])                
    
    # def init_optimal_outputlayer(self):
    #     with torch.no_grad():
    #         output_layer_weights = []
    #         output_layer_weights.append( torch.Tensor([[1]]))
    #         for i in range(self.num_hid_layers):
    #             output_layer_weights.append( torch.pow( torch.Tensor([[2]]), -2*(i+1) ) * torch.Tensor([[2,-4,2]]))
            
    #         self.layers[-1].weight.data = torch.cat(output_layer_weights, 1)
    #         self.layers[-1].bias.data = torch.Tensor([0])

    # def init_normal(self):
    #     for i in range(len(self.layers)):
    #         torch.nn.init.normal_(self.layers[i].weight, mean=0.0, std=1.0)
    #     with torch.no_grad():
    #         for i in range(0, len(self.layers) -1):
    #             self.layers[i].bias.data = torch.Tensor([0,0,0])
    #         self.layers[-1].bias.data = torch.Tensor([0])    
