import torch
import torch.nn as nn
import numpy as np
from divergence import Divergence  

device = "cpu"

class FisherLoss(nn.Module):
    def __init__(self, X, Z, epochs_max, dtype = torch.float64):
        super(FisherLoss, self).__init__()
        self.X = X
        self.Z = Z
        self.epochs_max = epochs_max

    def compute_fisher_loss(self):
        size = self.X.size()
        delta = 0.1  
        d = size[0]  
        shape = (d, d)
        BIM = torch.zeros(shape, device=device)  

        train_settings = {
            
            "sr": 5,
            "lr": 1e-4,
            "minor": 1e-3,
            "get_model": None,
            "debug": False
        }
        
        Z_ = [tensor.view(256, -1) for tensor in self.Z]
        Z = torch.cat(Z_, dim=1)
        print("X_np shape:", self.X.size())
        print("Z shape:", Z.shape)
        # print("Z_np shapes:", Z_np.size())
        X = self.X.view(256,-1)
        print("X shape: ", X.shape)
        BIM_diag = []
        for i in range(d-1):
            for j in range(i+1, d):
                origin = torch.cat((X, Z), dim=1)
                shifted = torch.cat((X, Z + 0.1), dim=1)  
                estimate = []

                for _ in range(1):
                    divergent = Divergence(origin, shifted, max_epochs=self.epochs_max, device=device)
                    estimate.append(2 * divergent.DivergenceApproximate(**train_settings))
                
                n_theta = np.max(estimate)
                B_diag = n_theta / (delta ** 2)

                if j==i:
                    BIM_diag.append(BIM_diag)
        loss = BIM.mean()
        
        return loss
