import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import os

from generator import Generator
from discriminator import Discriminator

image_size = 28
learning_rate = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 200
sample_interval = 1000
latent_dim = 100
n_classes = 10

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# return MNIST data_loader
def load_data():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return data_loader

def sample_image(n_row, batches_done, generator):
    
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "generate_result/%d.png" % batches_done, nrow=n_row, normalize=True)

def main(data_loader):
    adversarial_loss = nn.MSELoss()

    generator = Generator()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1,b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))


    '''
        Training
    '''
    os.makedirs(os.path.join(os.getcwd(), 'generate_result'), exist_ok=True )
    for epoch in range(n_epochs):
        print_d_loss = 0.0
        print_g_loss = 0.0
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad = False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad = False)

            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            '''
                Train Generator
            '''
            optimizer_G.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0,1,(batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
            gen_imgs = generator(z, gen_labels)

            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, real)

            g_loss.backward()
            optimizer_G.step()

            '''
                Train Discriminator
            '''
            optimizer_D.zero_grad()

            validity_real = discriminator(real_imgs, labels)
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            real_loss = adversarial_loss(validity_real, real)
            fake_loss = adversarial_loss(validity_fake, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(data_loader) + i
            if batches_done % sample_interval == 0:
                print('!!! save image %d.png to the generate_result folder !!!' % batches_done)
                sample_image(n_row = 10, batches_done=batches_done, generator=generator)

            print_d_loss += d_loss.item()
            print_g_loss += g_loss.item()

        print(
            "[Epoch %d/%d][D loss : %f][G loss : %f]"
            % (epoch, n_epochs, print_d_loss/len(data_loader), print_g_loss/len(data_loader))
        )

if __name__=="__main__":
    data_loader = load_data()
    main(data_loader)