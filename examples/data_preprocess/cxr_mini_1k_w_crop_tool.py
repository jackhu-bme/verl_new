import argparse
import os
import pandas as pd
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from verl.utils.hdfs_io import copy, makedirs

#todo: further check and refine!

def convert_row(row: dict, split_name: str) -> dict:
    seed = row["seed"]
    disease = row["disease"]
    exists = str(row["exists"]).strip().lower()
    question = f"Here is the X-ray of a single patient <image> of image index {seed}. You are an experienced radiologist. Does this patient have {disease}?"
    INSTRUCTION_FOLLOWING = (
        "You should answer in two steps: 1) First, conduct only one time tool_call of crop_image (need to pass the image index and crop coordinates, follow the format provided later)"
        "to get a clearer view of x-ray."
        "2) Second, Based on the original image and crooped image, think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST be put in \\boxed{}. Answer in English, using only 'yes' or 'no'. No other words. "
        "Example for image index -1: <tool_call>{\"name\": \"crop_image\", \"arguments\":{\"index\": -1, \"coordinates\": \"[100, 200, 120, 230]\"}}</tool_call>"
        "<think> I see signs of pneumonia in the lung fields. </think>. My answer: \\boxed{yes}"
    )
    prompt_text = question + " " + INSTRUCTION_FOLLOWING
    answer = "yes" if exists == "yes" else "no"

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
            "disease": disease,
            "dicom_id": row["dicom_id"],
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_parquet", default="/home/aiscuser/verl_new/cxr_mini_crop/full.parquet")
    parser.add_argument("--train_parquet", default="/home/aiscuser/verl_new/cxr_mini_crop/train.parquet")
    parser.add_argument("--test_parquet", default="/home/aiscuser/verl_new/cxr_mini_crop/test.parquet")
    parser.add_argument("--local_dir", default="~/data/cxr_mini_tool")
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
