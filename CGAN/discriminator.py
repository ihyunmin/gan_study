import torch
import torch.nn as nn
import numpy as np

img_shape = (1, 28, 28)
n_classes = 10

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear( n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,1),
        )

    def forward(self, img, labels):
        discriminator_input = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(discriminator_input)

        return validity