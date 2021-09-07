import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1), # output one value as fake or not
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        x = x.view(x.shape[0], -1)
        return  x