import torch
from tensorboardX.utils import make_grid
from torch import nn
import matplotlib.pyplot as plt

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)*0.7


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images,z_dim,device)
    fake_images = gen(noise)
    fake_pred = disc(torch.detach(fake_images))
    real_float = real.float()
    real_pred = disc(real_float)

    fake_loss = criterion(fake_pred,torch.zeros_like(fake_pred))
    real_loss = criterion(real_pred,torch.ones_like(real_pred))
    disc_loss = (fake_loss+ real_loss)/2
    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images,z_dim,device)
    fake_images = gen(noise)
    fake_pred = disc(fake_images)

    gen_loss = criterion(fake_pred,torch.ones_like(fake_pred))
    return gen_loss, fake_images


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.LeakyReLU(0.2,inplace=True)
    )

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
