import torch
from models.generator import Generator
from models.discriminator import Discriminator

G = Generator()
D = Discriminator()

z = torch.randn(8,100,1,1)

fake = G(z)
print(fake.shape)

out = D(fake)
print(out.shape)
