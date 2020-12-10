from class_dataset import MyDataSet
from class_model import MyModel
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector



def train_with_adam(model, criterion, trainset, batch_size, lr, epochs = 500, reg = 0, train_losses = [], norm_g = [], record_g = False , verbose=False):
    optimizer1 = optim.Adam(model.parameters(), lr)
  

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    for e in range(epochs):  
        model.train()
        for ind, data in enumerate(trainloader):
            inputs, labels = data['X'], data['y']

            def closure():
                optimizer1.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if (reg > 0):
                    reg_loss = 0
                    for i in range( len(model.layers)):
                        reg_loss += reg * torch.sum(torch.pow(model.layers[i].weight, 2))
                    loss += reg_loss
                loss.backward()

                if verbose:
                    print(f"loss = {loss}")
                if (ind==0):
                    train_losses.append(loss.unsqueeze(0))

                if (ind==0 and record_g):
                    all_gradients = [torch.flatten(model.layers[i].weight.grad.data) for i in range(len(model.layers))] + [torch.flatten(model.layers[i].bias.grad.data) for i in range(len(model.layers))]
                    g = torch.cat(all_gradients)
                    norm_g.append(torch.norm(g))

                return loss
            optimizer1.step(closure)

    return 



# train with LBFGS
def train_with_LBFGS(model, criterion, X_train, y_train, epochs = 100, reg = 0, train_loss=[], norm_g=[],record_g = 0, verbose = False):
    # last_loss=0

    # record gradients seperately
    if(record_g==2):
        for i in range(len(model.layers)):
            norm_g.append([])
            norm_g.append([])


    optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()),lr=1, max_iter=10000, tolerance_grad=0, tolerance_change=0, line_search_fn = 'strong_wolfe')
    # optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()),lr=1, line_search_fn = 'strong_wolfe')

    for e in range(epochs):  
        model.train()

        def closure():
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)    
            
            if (reg > 0):
                reg_loss = 0
                for i in range( len(model.layers)):
                    reg_loss += reg * torch.sum(torch.pow(model.layers[i].weight, 2))
                loss += reg_loss
            loss.backward()
            if verbose:
                print(f"loss = {loss}")
            l = loss.detach().unsqueeze(0)
            train_loss.append(l)
            if (record_g==1):
                with torch.no_grad():
                    all_gradients = [torch.flatten(model.layers[i].weight.grad.data) for i in range(len(model.layers))] + [torch.flatten(model.layers[i].bias.grad.data) for i in range(len(model.layers))]

                    g = torch.cat(all_gradients)
                    norm_g.append(torch.norm(g).unsqueeze(0))
            elif (record_g==2):
                with torch.no_grad():

                    for i in range(len(model.layers)):
                        norm_g[i].append(torch.norm(torch.flatten(model.layers[i].weight.grad.data)).unsqueeze(0))
                        norm_g[i+len(model.layers)].append(torch.norm(torch.flatten(model.layers[i].bias.grad.data)).unsqueeze(0))
            return loss

        optimizer.step(closure)

    return train_loss[-1]


def train_with_SGD(model, criterion, trainset, batch_size, lr, momentum = 0.9, weight_decay = 0.01, epochs = 10, reg = 0, train_losses = [], norm_g = [], record_g = False, verbose=False):
    optimizer1 = optim.SGD(model.parameters(), lr, momentum, weight_decay, nesterov=False)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    for e in range(epochs):  
        model.train()
        
        for ind, data in enumerate(trainloader):
            inputs, labels = data['X'], data['y']

            def closure():
                optimizer1.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)    

                if (reg > 0):
                    reg_loss = 0
                    for i in range( len(model.layers)):
                        reg_loss += reg * torch.sum(torch.pow(model.layers[i].weight, 2))
                    loss += reg_loss
                loss.backward()
                if verbose:
                    print(f"loss = {loss}")
                if (ind==0):
                    train_losses.append(loss.unsqueeze(0))

                if (ind==0 and record_g):
                    with torch.no_grad():
                        all_gradients = [torch.flatten(model.layers[i].weight.grad.data) for i in range(len(model.layers))] + [torch.flatten(model.layers[i].bias.grad.data) for i in range(len(model.layers))]
                        g = torch.cat(all_gradients)
                        norm_g.append(torch.norm(g).unsqueeze(0))

                return loss
            optimizer1.step(closure) #SGD

    return