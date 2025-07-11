import argparse
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from verl.utils.hdfs_io import copy, makedirs

# -------------------------
# Helper for image encoding
# -------------------------

# def image_to_bytes_dict(image_path: str):
#     """Read an image from disk and convert it to the HF datasets `bytes` dict format."""
#     with Image.open(image_path) as img:
#         return img

# # -------------------------
# # Per‑row conversion logic
# # -------------------------

# INSTRUCTION_FOLLOWING = (
#     "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
#     "The reasoning process MUST BE enclosed within <think> </think> tags. "
#     "The final answer MUST BE put in \\boxed{}."
# )

# def convert_row(row: dict, split_name: str):
#     """Convert a single row dict from full_df to verl format."""
#     seed = row["seed"]
#     disease = row["disease"]
#     exists = str(row["exists"]).strip().lower()

#     # Build prompt
#     question = f"Does this patient have {disease}?"
#     prompt_text = question + " " + INSTRUCTION_FOLLOWING

#     # yes / no answer
#     answer = "yes" if exists == "yes" else "no"

#     # Image bytes
#     images = [image_to_bytes_dict(row["img_256_path"])]

#     return {
#         "data_source": "cxr_crop",
#         "prompt": [
#             {
#                 "role": "user",
#                 "content": prompt_text,
#             }
#         ],
#         "images": images,
#         "ability": "chest_xray",
#         "reward_model": {"style": "rule", "ground_truth": answer},
#         "extra_info": {
#             "split": split_name,
#             "seed": seed,
#             "answer": answer,
#             "question": question,
#             "disease": disease,
#             "dicom_id": row["dicom_id"],
#         },
#     }


# updated functions for use

import io
import pyarrow as pa
import pandas as pd
from PIL import Image


def image_to_bytes_dict(image_path: str) -> dict:
    """Read an image and return a dict with raw bytes + metadata."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")                # 确保三通道
        width, height = img.size
        n_channels = len(img.getbands())        # e.g. 3
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")          # 或 JPEG
        img_bytes = buffer.getvalue()

    return {
        "bytes": pa.array([img_bytes], type=pa.binary())[0],  # PyArrow Binary
        "width": width,
        "height": height,
        "n_channels": n_channels,
        "format": "PNG",
    }

def convert_row(row: dict, split_name: str) -> dict:
    seed = row["seed"]
    disease = row["disease"]
    exists = str(row["exists"]).strip().lower()
    question = f"Here is the X-ray of a single patient <image>. You are a experienced radiologist. Does this patient have {disease}?"
    INSTRUCTION_FOLLOWING = (
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE put in \\boxed{}. Answer in English, using only 'yes' or 'no'."
    )
    prompt_text = question + " " + INSTRUCTION_FOLLOWING
    answer = "yes" if exists == "yes" else "no"

    # 关键：把 PIL Image 转成 bytes+meta dict
    # just path now to avoid bugs
    img_dict = {"image": "file://" + row["img_256_path"] }
    return {
        "data_source": "cxr_crop",
        "prompt": [
            {"role": "user", "content": prompt_text}
        ],
        "images": [img_dict],
        "ability": "chest_xray",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split_name,
            "index": seed,
            "answer": answer,
            "question": question,
            "disease": disease,
            "dicom_id": row["dicom_id"],
        },
    }

# # 假设 full_df 是个 pandas.DataFrame
# records = []
# for _, row in full_df.iterrows():
#     records.append(convert_row(row, split_name="train"))

# out_df = pd.DataFrame(records)

# # 写到 parquet
# out_df.to_parquet("output_with_images.parquet", index=False)









# -------------------------
# Main entry
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_parquet", default="/home/aiscuser/verl_new/cxr_mini_crop/full.parquet")
    parser.add_argument("--train_parquet", default="/home/aiscuser/verl_new/cxr_mini_crop/train.parquet")
    parser.add_argument("--test_parquet", default="/home/aiscuser/verl_new/cxr_mini_crop/test.parquet")
    parser.add_argument("--local_dir", default="~/data/cxr_mini")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--num_proc", type=int, default=os.cpu_count() // 2,
                        help="Number of parallel processes for image encoding")
    args = parser.parse_args()

    # Resolve user path shortcut (~)
    args.local_dir = os.path.expanduser(args.local_dir)

    # -------------------------
    # Load parquet files
    # -------------------------
    full_df = pd.read_parquet(args.full_parquet)

    # train / test seeds are stored inside extra_info
    train_seeds = (
        pd.read_parquet(args.train_parquet)["extra_info"].apply(lambda x: x["seed"]).tolist()
    )
    test_seeds = (
        pd.read_parquet(args.test_parquet)["extra_info"].apply(lambda x: x["seed"]).tolist()
    )

    # -------------------------
    # Process split helper
    # -------------------------
    def process_split(split_name: str, split_seeds: list):
        split_df = full_df[full_df["seed"].isin(split_seeds)]
        records = split_df.to_dict(orient="records")

        # Multiprocessing pool
        converter = partial(convert_row, split_name=split_name)
        split_data = []
        # for item in tqdm(records):
        #     res = converter(item)
        #     split_data.append(res)
        #     break  # debug only
        with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
            for item in tqdm(
                executor.map(converter, records, chunksize=8),
                total=len(records),
                desc=f"Processing {split_name} with {args.num_proc} procs",
            ):
                split_data.append(item)
        return split_data

    # -------------------------
    # Run conversion
    # -------------------------
    train_data = process_split("train", train_seeds)
    test_data = process_split("test", test_seeds)

    # -------------------------
    # Save parquet
    # -------------------------
    os.makedirs(args.local_dir, exist_ok=True)
    pd.DataFrame(train_data).to_parquet(os.path.join(args.local_dir, "train.parquet"))
    pd.DataFrame(test_data).to_parquet(os.path.join(args.local_dir, "test.parquet"))

    # -------------------------
    # Optional HDFS upload
    # -------------------------
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)

    print("✅ Conversion finished. Train / test parquet saved to", args.local_dir)
