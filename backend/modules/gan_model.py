import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, noise_dim, data_dim):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(noise_dim,128),
            nn.ReLU(),

            nn.Linear(128,256),
            nn.ReLU(),

            nn.Linear(256,data_dim),

            nn.Tanh()
        )

    def forward(self,x):

        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self,data_dim):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(data_dim,256),
            nn.LeakyReLU(0.2),

            nn.Linear(256,128),
            nn.LeakyReLU(0.2),

            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):

        return self.model(x)
