import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(3, 6, 4),
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            # ((W - F) + 2*P) / S + 1
            # ((32 - 4) + 2*0) / 2 + 1 = 15
            nn.Conv2d(6, 9, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # ((15 - 5) + 2*0) / 2 + 1 = 6
        )
        self.linear = nn.Sequential(
            nn.Linear(5*5*9, 80),
            nn.ReLU(),
            nn.Linear(80, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 5*5*9)
        x = self.linear(x)
        return x