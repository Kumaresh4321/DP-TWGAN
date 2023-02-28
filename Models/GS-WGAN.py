import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GSWGAN(object):
    def __init__(self, generator, discriminator, noise_function):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_function = noise_function

    def train(self, data, epochs=100, n_critics=5, batch_size=128,
              learning_rate=1e-4, sigma=None, weight_clip=0.1):
        generator_solver = optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )
        discriminator_solver = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )

        # add hooks to introduce noise to gradient for differential privacy
        if sigma is not None:
            for parameter in self.discriminator.parameters():
                parameter.register_hook(
                    lambda grad: grad + (1 / batch_size) * sigma
                    * torch.randn(parameter.shape)
                )

        # There is a batch for each critic (discriminator training iteration),
        # so each epoch is epoch_length iterations, and the total number of
        # iterations is the number of epochs times the length of each epoch.
        epoch_length = len(data) / (n_critics * batch_size)
        n_iters = int(epochs * epoch_length)
        for iteration in range(n_iters):
            for _ in range(n_critics):
                # Sample real data
                rand_perm = torch.randperm(data.size(0))
                samples = data[rand_perm[:batch_size]]
                real_sample = samples.view(batch_size, -1)

                # Sample fake data
                noise = self.noise_function(batch_size, self.generator.latent_dim)
                fake_sample = self.generator(noise)

                # Score data
                discriminator_real = self.discriminator(real_sample)
                discriminator_fake = self.discriminator(fake_sample)

                # Calculate discriminator loss
                # Discriminator wants to assign a high score to real data
                # and a low score to fake data
                discriminator_loss = -(
                    torch.mean(discriminator_real) -
                    torch.mean(discriminator_fake)
                )

                discriminator_loss.backward()
                discriminator_solver.step()

                # Weight clipping for privacy guarantee
                for param in self.discriminator.parameters():
                    param.data.clamp_(-weight_clip, weight_clip)

                # Reset gradient
                self.generator
