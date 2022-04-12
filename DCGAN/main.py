from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from generator import Generator
from discriminator import Discriminator
import os
from torchvision.utils import save_image

# set Seed
# manualSeed = 999
manualSeed = random.randint(1,10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "dataset"
workers = 1
batch_size = 128
image_size = 64
nc = 3 # Number of channels
nz = 100
ndf = 64 # Size of feature maps in discrimintor
num_epochs = 5 # epoch
lr = 0.0002 # learning rate
beta1 = 0.5 # Beta1 hyper-parameter for Adam optimizers
ngpu = 1 # Number of GPUs
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >0) else "cpu")

def data_load():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    dataset = datasets.ImageFolder(root=dataroot, transform=transform)
    dataloader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    dataloader = data_load()
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    # print(netG)
    # print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.rand(64, nz, 1,1, device=device)
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Traning Loop!")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            '''
                1. update D net : maximize log(D(x)) + log(1 - D(G(x)))
            '''
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            D_loss_real = criterion(output, label)
            D_loss_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            D_loss_fake = criterion(output, label)
            D_loss_fake.backward()
            D_G_z1 = output.mean().item()
            D_loss = D_loss_real + D_loss_fake
            optimizerD.step()

            '''
                2. update G net : maximize log(D(G(x)))
            '''
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            G_loss = criterion(output, label)
            G_loss.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())

            if (iters%500 == 0) or ((epoch == num_epochs -1) and (i == len(dataloader)- 1)):
                save_image(fake.data[:25], os.path.join(os.getcwd(), 'generate_result', f"{iters}.png") , nrow=5, normalize=True)
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake ,padding=2, normalize=True))

            iters +=1
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminior Loss During Training")
    plt.plot(G_losses, label='G')
    plt.plot(D_losses, label='D')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generating Images")
    for i in range(len(img_list)):
        plt.axis("off")
        # plt.imshow(np.transpose(img_list[i],(1,2,0)))
        plt.imsave('generate_result/savefig_default' + str(i) +'.png', np.transpose(img_list[i],(1,2,0)).numpy().squeeze())
        # plt.show()
        # plt.savefig('savefig_default' + str(i) +'.png')
    plt.imshow(np.transpose(img_list[-1], (1,2,0)))
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())

if __name__=="__main__":
    main()
