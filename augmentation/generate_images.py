import torch
import yaml
import os

from torchvision.utils import save_image

from models.sgan import SGAN


def generate_samples(
    checkpoint_path,
    num_images=1000
):

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SGAN(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)

    generator = model.generator
    generator.eval()

    latent_dim = config["generator"]["latent_dim"]

    output_dir = config["dataset"]["generated_path"]

    os.makedirs(output_dir, exist_ok=True)

    batch_size = 64

    generated = 0

    with torch.no_grad():

        while generated < num_images:

            z = torch.randn(
                batch_size,
                latent_dim,
                1,
                1,
                device=device
            )

            fake_images = generator(z)

            for img in fake_images:

                path = os.path.join(
                    output_dir,
                    f"gen_{generated}.png"
                )

                save_image(
                    img,
                    path,
                    normalize=True
                )

                generated += 1

                if generated >= num_images:
                    break

    print(f"{generated} images generated.")


if __name__ == "__main__":

    generate_samples(
        checkpoint_path="outputs/checkpoints/model.pt",
        num_images=1000
    )
