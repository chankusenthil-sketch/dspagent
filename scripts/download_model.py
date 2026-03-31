from huggingface_hub import snapshot_download
import os
import sys

import argparse

# Preset models (use --preset flag to select)
MODEL_PRESETS = {
    "mistral": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "local_dir": "Mistral-7B-Instruct-v0.3",
        "ignore_patterns": ["*.gguf", "*.ggml", "consolidated*"],
    },
    "opt": {
        "repo_id": "facebook/opt-6.7b",
        "local_dir": "opt-6.7b",
        "ignore_patterns": [],
    },
    "llama2": {
        "repo_id": "meta-llama/Llama-2-13b-hf",
        "local_dir": "Llama-2-13b-hf",
        "ignore_patterns": [],
    },
    "llama3": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": "Llama-3.1-8B-Instruct",
        "ignore_patterns": ["*.gguf", "*.ggml", "original/*"],
    },
}

MODEL_ID_DEFAULT = "mistralai/Mistral-7B-Instruct-v0.3"


def parse_args():
    p = argparse.ArgumentParser(description="Download HuggingFace models for the DSP Agent LLM server.")
    p.add_argument("model", nargs="?", default=None, help="Hugging Face model repo id (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
    p.add_argument("--preset", choices=list(MODEL_PRESETS.keys()), default=None,
                   help="Use a preset model config: " + ", ".join(f"{k} ({v['repo_id']})" for k, v in MODEL_PRESETS.items()))
    p.add_argument("--out", default=None, help="Output directory for model snapshot")
    p.add_argument("--list", action="store_true", help="List available presets and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available presets:")
        for name, cfg in MODEL_PRESETS.items():
            print(f"  {name:10s}  {cfg['repo_id']}")
        sys.exit(0)

    # Resolve model config
    ignore_patterns = []
    if args.preset:
        preset = MODEL_PRESETS[args.preset]
        model_id = preset["repo_id"]
        ignore_patterns = preset["ignore_patterns"]
        default_dir = preset["local_dir"]
    elif args.model:
        model_id = args.model
        default_dir = model_id.replace("/", "-")
    else:
        # Default to Mistral
        preset = MODEL_PRESETS["mistral"]
        model_id = preset["repo_id"]
        ignore_patterns = preset["ignore_patterns"]
        default_dir = preset["local_dir"]

    outdir = args.out
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "..", "models", default_dir)

    token_path = os.path.join(os.path.dirname(__file__), "..", "hf_token.txt")
    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()

    print(f"Downloading {model_id} to {outdir} ...")
    if ignore_patterns:
        print(f"  Ignoring patterns: {ignore_patterns}")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=outdir,
        token=token,
        ignore_patterns=ignore_patterns or None,
    )
    print("Model snapshot downloaded to:", path)


if __name__ == "__main__":
    main()
