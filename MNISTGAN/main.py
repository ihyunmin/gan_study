from discriminator import Discriminator
from generator import Generator
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time
import os

latent_dim = 100

def load_data():
    transforms_train = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms_train)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    return dataloader

def main(dataloader):
    generator = Generator()
    discriminator = Discriminator()
    generator.cuda()
    discriminator.cuda()

    adversarial_loss = nn.BCELoss()
    adversarial_loss.cuda()
    lr = 0.002
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    n_epochs = 2000
    sample_interval = 2000
    start_time = time.time()

    os.makedirs(os.path.join(os.getcwd(), 'generate_result'), exist_ok=True )
    for epoch in range(n_epochs):
        for i, (imgs, _ ) in enumerate(dataloader):
            real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0)
            fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0)

            real_imgs = imgs.cuda()

            ''' Generator learning '''
            optimizer_G.zero_grad()
            # latent vector generate
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()
            generated_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(generated_imgs), real)
            g_loss.backward()
            optimizer_G.step()

            ''' Discriminator learning '''
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            done = epoch * len(dataloader) + i
            if done % sample_interval == 0:
                save_image(generated_imgs.data[:25], os.path.join(os.getcwd(), 'generate_result', f"{done}.png") , nrow=5, normalize=True)

        print(f'[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss : {g_loss.item():.6f}] [Elapsed time: {time.time()-start_time:.2f}s]')


if __name__ == "__main__":
    dataloader = load_data()
    main(dataloader)

