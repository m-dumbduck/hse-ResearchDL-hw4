import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--repo-id", required=True, help="HF repo id")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "generator_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'generator_state_dict' key")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint must contain 'config' key")

    generator = instantiate(checkpoint["config"].generator.model)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    generator.push_to_hub(
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
