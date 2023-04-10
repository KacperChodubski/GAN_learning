import torch
import torch.nn as nn
from typing import Iterator
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tranforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import Descriminator,  Generator


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = Descriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = tranforms.Compose([
    tranforms.ToTensor(),
    tranforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(),lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
writer_fake = SummaryWriter(log_dir="/Users/mojskarb/my_projects/GAN/runs/GAN_MNIST/fake")
writer_real = SummaryWriter(log_dir="/Users/mojskarb/my_projects/GAN/runs/GAN_MNIST/real")
criterion = nn.BCELoss()
step = 0

for epoch in range(num_epochs):
    for batch_id, (real, _) in enumerate(dataset_loader):
        real = real.view(-1, 28*28*1).to(device)
        #batch_size = real.shape[0]

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_id == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"LossD: {lossD:.4f}, LossG: {lossG:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

