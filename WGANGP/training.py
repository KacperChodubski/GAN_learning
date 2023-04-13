import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights
from utils import gradient_penalty
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 10
FEATURES_DISC = 32
FEATURES_GEN = 32
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

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
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_fake = SummaryWriter("logs/fake")
writer_real = SummaryWriter("logs/real")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.to(device)

        for _ in range(CRITIC_ITERATIONS):
        ### critic training
            noise = torch.randn(real.shape[0], Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        output = critic(fake).view(-1)
        loss_gen = -(torch.mean(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx}/{len(train_loader)}  | "
                f"LossC: {loss_critic:.4f}, LossG: {loss_gen:.4f}")
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


        
