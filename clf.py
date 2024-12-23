import torch
import torch.nn as nn
import torchelie.nn as tnn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
            

class L2Pool(nn.Module):
    def forward(self, x):
        return x.norm(dim=(2,3))


def ClassLoss(in_dim, hidden_dim, num_classes):
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(in_dim, hidden_dim, 1),
        ResBlock(hidden_dim),
        ResBlock(hidden_dim),
        ResBlock(hidden_dim),
        #nn.AdaptiveAvgPool2d(1),
        L2Pool(),
        nn.Linear(hidden_dim, num_classes),
        tnn.Reshape(num_classes),
    )

