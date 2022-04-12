import torch
import torch.nn as nn
import numpy as np

latent_dim = 100
n_classes = 10
# img_shape // channel, width, height
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normailze=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normailze:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normailze=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img