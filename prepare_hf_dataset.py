"""
Convert the preprocessed parquet to HuggingFace Dataset format.

Usage:
    python prepare_hf_dataset.py                  # just convert locally
    python prepare_hf_dataset.py --push            # push to Hub
    python prepare_hf_dataset.py --push --token hf_xxx  # push with explicit token

Output:
    kodcode_processed/hf_dataset/  — HF Dataset in arrow format
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_from_disk

PARQUET_PATH = Path("kodcode_processed/kodcode-lfm2.5.parquet")
OUTPUT_DIR = Path("kodcode_processed/hf_dataset")
HF_REPO = "enseven/kodcode-lfm2.5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--token", type=str, default=None, help="HF token for push")
    args = parser.parse_args()

    print(f"Loading parquet from {PARQUET_PATH} ...", flush=True)
    ds = Dataset.from_parquet(str(PARQUET_PATH))
    print(f"Dataset: {len(ds)} rows, columns: {ds.column_names}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR} ...", flush=True)
    ds.save_to_disk(str(OUTPUT_DIR))

    # Calculate size
    total_size = sum(
        os.path.getsize(p) for p in OUTPUT_DIR.rglob("*") if p.is_file()
    )
    print(f"  {total_size / 1024**3:.2f} GB", flush=True)

    if args.push:
        token = args.token or os.environ.get("HF_TOKEN")
        ds.push_to_hub(HF_REPO, private=True, token=token)
        print(f"Pushed to https://huggingface.co/datasets/{HF_REPO}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
