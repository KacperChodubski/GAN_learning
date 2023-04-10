import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARING_RATE_GEN = 2e-4
LEARING_RATE_DISC = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 40
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = tf.Compose([
    tf.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    tf.ToTensor(),
    tf.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
])

#train_dataset = datasets.MNIST("./dataset", train=True, transform=transforms, download=True)
#test_dataset = datasets.MNIST("./dataset", train=False, transform=transforms, download=False)


path = os.path.join(os.path.dirname(__file__), "dataset", "Celeb")
train_dataset = datasets.ImageFolder(root=path, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

gen_optim = optim.Adam(gen.parameters(), lr=LEARING_RATE_GEN, betas=(0.5, 0.999))
disc_optim = optim.Adam(disc.parameters(), lr=LEARING_RATE_DISC, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(128, Z_DIM, 1, 1).to(device)

writer_fake = SummaryWriter("logs/fake")
writer_real = SummaryWriter("logs/real")
step = 0


for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ### disc train
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        disc_optim.step()

        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()

        if batch_idx % 400 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] \ "
                f"LossD: {loss_disc:.4f}, LossG: {loss_gen:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1


        
