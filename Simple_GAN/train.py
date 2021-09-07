import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from generator import Generator
from discriminator import Discriminator
from dataloader import Dataload


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 64
LR = 3e-4
Z_DIM = 64
IMAGE_DIM = 1 * 28 * 28 # MNIST

# Dataset
loader = Dataload(batch_size=BATCH_SIZE)

# Model & Optimizer & Loss Initializations
disc = Discriminator(IMAGE_DIM).to(DEVICE)
gen = Generator(Z_DIM, IMAGE_DIM).to(DEVICE)
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(DEVICE)

optim_disc = optim.Adam(disc.parameters(), lr=LR)
optim_gen = optim.Adam(gen.parameters(), lr=LR)

criterion = nn.BCELoss()

# Tensorboard
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0


# Training
for epoch in range(EPOCHS):
    for idx, (real, _) in enumerate(loader):
        real = real.view(-1, 28*28).to(DEVICE)
        batch_size = real.shape[0]

        # Train Discriminator
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise)

        disc_real = disc(real)
        D_loss_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach())
        D_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        D_loss = (D_loss_real + D_loss_fake) / 2
        disc.zero_grad()
        D_loss.backward()
        optim_disc.step()

        # Train Generator
        output = disc(fake)

        G_loss = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        G_loss.backward()
        optim_gen.step()

        if idx == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} \n Discriminator Loss: {D_loss:.5f}, Generator Loss: {G_loss:.5f}")

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)

                step += 1
