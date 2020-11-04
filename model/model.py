import torch.nn as nn
import torch.nn.functional as F
from moGAN_git.base import BaseModel
from moGAN_git.utils.gan_utils import get_discriminator_block, get_generator_block


class moGanGenerator(BaseModel):

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            # nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)


    def get_gen(self):
        return self.gen

class moGanDiscriminator(BaseModel):

    def __init__(self, im_dim=784, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.disc(image)


    def get_disc(self):
        return self.disc

# class Discriminator(nn.Module):
#
#     def __init__(self, im_dim=784, hidden_dim=128):
#         super().__init__()
#         self.disc = nn.Sequential(
#             get_discriminator_block(im_dim, hidden_dim * 4),
#             get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
#             get_discriminator_block(hidden_dim * 2, hidden_dim),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, image):
#         return self.disc(image)
#
#
#     def get_disc(self):
#         return self.disc


##Example class from templete
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
