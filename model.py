import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(228, int(1e2)),
            nn.LeakyReLU(),
            nn.Linear(int(1e2), 10),
            nn.LeakyReLU(),
            nn.Linear(10,1))

    def forward(self, x):
        return self.model(x)