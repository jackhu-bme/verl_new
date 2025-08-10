import argparse
from pathlib import Path
import json
import base64
import io

import pyarrow.parquet as pq
from PIL import Image
from textwrap import indent

from io import BytesIO

import os

"""
Parquet Sample Comparator
-------------------------
Quickly inspect and compare one sample from two parquet datasets (e.g. Geometry3K vs. CXR).
It prints key fields and tries to decode & save the first image of each sample.
Supports three image storage patterns used by HF‑style datasets:
  • HF bytes‑dict   : {"bytes": {"0":137, ...}} or {"bytes": {0:137, ...}}
  • Raw bytes       : the field itself is a bytes object (pyarrow “binary” column)
  • Base64 string   : "data:image/png;base64,..."
"""

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_row(path: str, idx: int = 0):
    """Load a single sample (row idx) from parquet as python dict."""
    table = pq.read_table(path).slice(idx, 1)
    return table.to_pylist()[0]




from io import BytesIO




def summarize(sample: dict, tag: str, out_dir: Path):
    print(f"\n===== {tag} Sample =====")
    for key in ("data_source", "ability", "prompt", "reward_model", "extra_info"):
        if key in sample:
            val = sample[key]
            if key == "prompt" and isinstance(val, list):
                val = val[0]  # only first turn for brevity
            print(f"{key}:\n{indent(json.dumps(val, ensure_ascii=False, indent=2), '  ')}")

    if "images" in sample and sample["images"]:
        byte_dict = sample["images"][0]["bytes"]
        img_bytes = byte_dict
        # img_bytes = bytes([byte_dict[str(k)] for k in range(len(byte_dict))])
        image = Image.open(io.BytesIO(img_bytes))
        out_path = os.path.join(out_dir, f"{tag.lower()}_image.png")
        image.save(out_path)
        print(f"[{tag}] image saved to:", out_path)


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare one sample between two HF‑style parquet datasets")
    parser.add_argument("--geo_parquet", required=True, help="Path to Geometry3K parquet")
    parser.add_argument("--cxr_parquet", required=True, help="Path to CXR parquet")
    parser.add_argument("--idx", type=int, default=1, help="Row index to inspect")
    parser.add_argument("--out_dir", default="sample_outputs", help="Folder to write decoded images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geo_sample = load_row(args.geo_parquet, args.idx)
    # cxr_sample = load_row(args.cxr_parquet, args.idx)

    summarize(geo_sample, "Geometry3K", out_dir)
    # summarize(cxr_sample, "CXR", out_dir)

    print("\n🔍 Comparison complete – check printed fields and saved images.")
