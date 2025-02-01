from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_id", type=str, help="model_id")
    parser.add_argument("--cache_dir", type=str, default="", help="cache_dir")
    args = parser.parse_args()

    # download tokenizer
    AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir
    )

    print("Done: tokenizer")

    # download weight
    AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        cache_dir=args.cache_dir,
        device_map="auto", 
        torch_dtype = torch.bfloat16,
    )

    print("Done: weight")