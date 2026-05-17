import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=64, output_channels=3, output_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Layer 2
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # Layer 3
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # Output layer
            nn.Linear(1024, output_channels * output_size * output_size),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.model(z)
        # Reshape into an image
        output = output.view(-1, 3, self.output_size, self.output_size)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, input_size=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_channels * input_size * input_size, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # Flatten the image
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


if __name__ == "__main__":
    gen = Generator()
    disc = Discriminator()

    gen.eval()
    disc.eval()

    z = torch.randn(1, 64)

    # Generate a fake image
    fake_image = gen(z)
    print(f"Generator output shape: {fake_image.shape}")

    # Discriminator scores the fake image
    score = disc(fake_image)
    print(f"Discriminator score: {score.item():.4f}")

    print("Both models working correctly")