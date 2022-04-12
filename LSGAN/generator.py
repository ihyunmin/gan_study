import torch
import torch.nn as nn

# latent, noise vector dimension
latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024, 1*28*28),
            nn.Tanh()   # why does he use tanh()?
        )

    def forward(self, x):
        # x is noise vector
        img = self.model(x)
        img = img.view(img.size(0), 1, 28, 28)
        return img