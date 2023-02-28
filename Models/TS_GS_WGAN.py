import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import grad as torch_grad

class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        out = self.bn1(self.fc1(z))
        out = nn.LeakyReLU(0.2)(out)
        out = self.bn2(self.fc2(out))
        out = nn.LeakyReLU(0.2)(out)
        out = self.fc3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.bn1(self.fc1(x))
        out = nn.LeakyReLU(0.2)(out)
        out = self.bn2(self.fc2(out))
        out = nn.LeakyReLU(0.2)(out)
        out = self.fc3(out)
        return out.view(-1)

class TimeSeriesDataset(data.Dataset):
    def __init__(self, data):
        super(TimeSeriesDataset, self).__init__()
        self.data = torch.Tensor(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class GSWGAN(object):
    def __init__(self, generator, discriminator, data, noise_size=100,
                 batch_size=128, lr=0.0001, n_critic=5, clip_value=0.01,
                 gradient_penalty_weight=10):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.noise_size = noise_size
        self.batch_size = batch_size
        self.lr = lr
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.gradient_penalty_weight = gradient_penalty_weight

        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates.requires_grad = True

        disc_interpolates = self.discriminator(interpolates)

        gradients = torch_grad(outputs=disc_interpolates, inputs=interpolates,
                               grad_outputs=torch.ones(disc_interpolates.size()),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(self.batch_size, -1)

        gradient_penalty = (            (gradients.norm(2, dim=1) - 1) ** 2
        ).mean() * self.gradient_penalty_weight

        return gradient_penalty

    def train(self, epochs):
        dataset = TimeSeriesDataset(self.data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                real_data = data
                self.optimizer_d.zero_grad()

                # Train the discriminator for n_critic iterations
                for _ in range(self.n_critic):
                    noise = torch.randn(self.batch_size, self.noise_size)
                    fake_data = self.generator(noise).detach()
                    d_loss = (
                        -torch.mean(self.discriminator(real_data))
                        + torch.mean(self.discriminator(fake_data))
                    )
                    gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)
                    d_loss += gradient_penalty
                    d_loss.backward()
                    self.optimizer_d.step()
                    # Clip discriminator weights
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                # Train the generator
                self.optimizer_g.zero_grad()
                noise = torch.randn(self.batch_size, self.noise_size)
                fake_data = self.generator(noise)
                g_loss = -torch.mean(self.discriminator(fake_data))
                g_loss.backward()
                self.optimizer_g.step()

                if i % 10 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (
                            epoch,
                            epochs,
                            i,
                            len(dataloader),
                            d_loss.item(),
                            g_loss.item(),
                        )
                    )

           
