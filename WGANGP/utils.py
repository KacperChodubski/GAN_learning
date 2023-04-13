import torch
import torch.nn as nn
import os

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(critic, generator, optimizer_critic, optimizer_generator):
    print("=> Saving checkpoint")

    model_binaries_path = os.path.join(os.path.dirname(__file__), 'trained_models', 'binaries')

    training_state_critic = {
        'state_dict': critic.state_dict(),
        'optimizer_state': optimizer_critic.state_dict(),
    }
    critic_name = f'model_critic.pth'
    torch.save(training_state_critic, os.path.join(model_binaries_path, critic_name))

    training_state_generator = {
        'state_dict': generator.state_dict(),
        'optimizer_state': optimizer_generator.state_dict(),
    }
    generator_name = f'model_generator.pth'
    torch.save(training_state_generator, os.path.join(model_binaries_path, generator_name))


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])