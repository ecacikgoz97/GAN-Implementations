import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(), # make sure output is in range [-1, 1]
        )

    def forward(self, x):
        x = self.generator(x)
        x = x.view(x.shape[0], -1)
        return x