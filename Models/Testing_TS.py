import numpy as np
import pandas as pd
import torch
from TS_GS_WGAN import Generator, Discriminator, TimeSeriesDataset, GSWGAN

df = pd.read_csv('C:\Semester 8 Project\ChatGPT\ADANIGREEN_minute_data_with_indicators.csv\ADANIGREEN_minute_data_with_indicators.csv', header=None)

data = df.values


input_size = 100
output_size = 2
hidden_size = 50
noise_size = 100
batch_size = 50
lr = 0.00001
n_critic = 5
clip_value = 0.01
gradient_penalty_weight = 10
epochs = 50

generator = Generator(input_size=noise_size, output_size=output_size, hidden_size=hidden_size)
discriminator = Discriminator(input_size=output_size, hidden_size=hidden_size)

dataset = TimeSeriesDataset(data=data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

gs_wgan = GSWGAN(generator=generator, discriminator=discriminator, data=dataloader,
                 noise_size=noise_size, batch_size=batch_size, lr=lr, n_critic=n_critic,
                 clip_value=clip_value, gradient_penalty_weight=gradient_penalty_weight)

gs_wgan.train(epochs=epochs)

synthetic_data = generator(torch.randn(10, noise_size))
