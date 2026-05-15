import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", required=True, help="Path to export")
    parser.add_argument("--repo-id", required=True, help="HF repo id")
    parser.add_argument("--repo-path", required=True, help="HF repo path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    if "generator_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'generator_state_dict' key")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["generator_state_dict"], output_path)
    hf_api = HfApi()
    hf_api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    hf_api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=args.repo_path,
        repo_id=args.repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    main()
