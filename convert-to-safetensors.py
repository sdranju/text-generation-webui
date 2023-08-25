'''

Converts a transformers model to safetensors format and shards it.

This makes it faster to load (because of safetensors) and lowers its RAM usage
while loading (because of sharding).

Based on the original script by 81300:

https://gist.github.com/81300/fe5b08bff1cba45296a829b9d6b0f303

'''

import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
    parser.add_argument('MODEL', type=str, default=None, nargs='?', help="Path to the input model.")
    parser.add_argument('--output', type=str, default=None, help='Path to the output folder (default: models/{model_name}_safetensors).')
    parser.add_argument("--max-shard-size", type=str, default=None, help="Maximum size of a shard in GB or MB, i.e. '2GB' (default: %(default)s).")
    parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
    args = parser.parse_args()

    model_path = args.MODEL

    if model_path is None:
        print("Error: Please specify the model you'd like to convert (e.g. 'python convert-to-safetensors.py "
              "models/Deci_DeciCoder-1b').")
        sys.exit()

    path = Path(model_path)
    model_name = path.name
    max_shard_size = args.max_shard_size

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(path)

    out_folder = args.output or Path(f"models/{model_name}_safetensors")
    if max_shard_size is not None:
        print(f"Saving the converted model to {out_folder} with a maximum shard size of {args.max_shard_size}...")
        model.save_pretrained(out_folder, max_shard_size=args.max_shard_size, safe_serialization=True)
    else:
        print(f"Saving the converted model to {out_folder}...")
        model.save_pretrained(out_folder, safe_serialization=True)

    tokenizer.save_pretrained(out_folder)
