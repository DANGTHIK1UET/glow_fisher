import numpy as np
import torch
from torch import nn, optim
from multiprocessing import Process, Manager
from tqdm import tqdm
import time, copy
import matplotlib.pyplot as plt

def Donsker_Varadhan_loss(_model, x, y):
    T_x = _model(x)
    T_y = _model(y)
    return -(torch.mean(T_x) - torch.log(torch.mean(torch.exp(T_y))))

class Divergence(nn.Module):
    def __init__(self, x_pool, y_pool, max_epochs, device):
        super(Divergence,self).__init__()
        self.x_pool = x_pool
        self.y_pool = y_pool
        self.max_epochs = max_epochs
        self.to(device)

    def DivergenceApproximate(self, batch_size=2, get_model = None, sr=1, lr=1e-2, debug=False, minor=5e-3, loss=Donsker_Varadhan_loss):
    # Create a model if not specified
        estimates = []    # Store losses
        window = 6
        start = 0
        end = batch_size
        TENSOR_TYPE = torch.float32
        x_pool = torch.tensor(self.x_pool, dtype=TENSOR_TYPE, requires_grad=True)
        y_pool = torch.tensor(self.y_pool, dtype=TENSOR_TYPE, requires_grad=True)
        if get_model is None:
            d = (x_pool).shape[1]
            if (int(d * sr) < 1):
                raise Exception("The number of units in the hidden layer must be greater than zero!")
            model = nn.Sequential( 
                nn.Linear(d, int(d * sr)),
                nn.ReLU(),
                nn.Linear(int(d * sr), 1))
        else:
            model = get_model()

        # Create an optimizer
        opt = optim.Adam(model.parameters(), lr=lr)
        # Calculate batch_size if necessary
        batch_size = int(max(100, x_pool.shape[0] // 10 if batch_size is None else batch_size))
        batch_size = min(batch_size, x_pool.shape[0])
        batch_size = 2
        
        # Train the statistics network to estimate the KL divergence
        for epoch in tqdm(range(self.max_epochs)) if debug else range(self.max_epochs):
            x_pool = x_pool[torch.randperm(x_pool.size()[0])]
            y_pool = y_pool[torch.randperm(y_pool.size()[0])]
            # SGD learn a whole epoch
            opt.zero_grad()
            for start in range(0, x_pool.shape[0], batch_size):
                end = min(x_pool.shape[0], start + batch_size)
                x = x_pool[start:end]
                y = y_pool[start:end]
                # Evaluate gradients
                loss_value = loss(model, x, y)
                loss_value.backward(retain_graph = True)
                # Apply them
                opt.step()

            # Calculate and store the loss
            epoch_loss = 0.0
            with torch.no_grad():
                epoch_loss = loss(model, x_pool, y_pool)
            estimates.append(epoch_loss.cpu())

            # Check if diverge
            if np.isnan(estimates[-1]):
                start = -1  # Recreate the model, start from a new random point
                estimates[-1] = 0.0
            # Check stop condition
            elif epoch >= window and estimates[-1] != 0:
                std = np.std(estimates[-window:])
                if std / np.abs(estimates[-1]) < minor:
                    # If converge to a negative value, try again
                    if estimates[-1] > 0:
                        if debug:
                            print("Converge to a negative value.")
                        start = -1  # Recreate the model, start from a new random point
                    else:
                        # Done
                        break

            if start == -1:
                # Recreate the model, start from a new random point
                # model = nn.Sequential(
                #     nn.LazyLinear(int(d * sr)),
                #     nn.ReLU(),
                #     nn.LazyLinear(1))

                # model = reinitialize_weight(model)
                if get_model is None:
                    model = nn.Sequential(
                        nn.Linear(d, int(d * sr)),
                        nn.ReLU(),
                        nn.Linear(int(d * sr), 1))
                else:
                    model = get_model()
                opt = optim.Adam(model.parameters(), lr=lr)

            # Begin new epoch
            start = 0
            end = batch_size

        if debug:
            plt.plot(estimates)
            plt.show()
        # if epoch == self.max_epochs - 1:
        #     if debug:
        #         pass
                # print("Lim reached!", -np.min(estimates[-window:]))
            # estimates = most_stable_part(estimates, window=10)
        if len(estimates) == 0:
            return 0.0
        return max(-np.min(estimates), 0.0)
    

