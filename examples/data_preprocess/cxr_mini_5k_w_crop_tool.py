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
    question = f"Here is the X-ray of a single patient <image> of image index {seed}. Does this patient have {disease}?"
    INSTRUCTION_FOLLOWING = (
       """Now only conduct the [stage 1] step, you need to do the tool_call by returning a json object with the function name and arguments within <tool_call><tool_call> XML tags:
       <tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n<tool_call>.
       After this, wait for the tool_call results. More instructions of stage 2 will be provided in tool_call results. Follow them only after you finish stage 1 and recieve the tool_call results."""
    )
    prompt_text = question + " " + INSTRUCTION_FOLLOWING
    answer = "yes" if exists == "yes" else "no"

    img_dict = {"image": "file://" + row["img_256_path"] }

    return {
        "data_source": "cxr_crop",
        "prompt": [
            {   "role": "system", 
                "content": "You are an experienced radiologist. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should answer in two steps: 1) [stage 1] First, conduct only one time tool_call of crop_image (need to pass the image index and crop coordinates, follow the format provided later)"
                            "2) [stage 2] Second, Based on the original image and crooped image, think about the reasoning process as an internal monologue and then provide the final answer. "
                            "The reasoning process MUST BE enclosed within <think> </think> tags. "
                            "The final answer MUST be put in \\boxed{}. Answer in English, using only 'yes' or 'no'. No other words. "
                            "Example whole process for image index -1: "
                            "<tool_call>\n{'name': 'crop_image', 'arguments':{'index': -1, 'coordinates': '[106, 163, 128, 237]'}}</tool_call>"
                            f"you should repeat the image index: {seed} for the index parameter, or you fail totally."
                            "remember to use <tool_call>...</tool_call> to surround your tool call json object, or it can not be recognized. "
                            f"these coordinates are just examples, you should choose based on the need for your final diagnosis of disease: {disease}. "
                            "then the coordniates should be in the format of [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
                            "the croppind region is based on your coorinates of the original image, "
                            f"To get a clearer view of x-ray, make this only one cropping region related to your final diagnosis of disease: {disease}. "
                            "then the tool call will give you some results and hints. Only after this, you can say:"
                            "Example 1 (positive situations): <think>In both the original image and the cropped region, I see consistent patchy opacities and consolidation in the lung fields, confirming pneumonia. </think>. My answer: \\boxed{yes}.  "
                            "Example 2 (negative situations): <think>Neither the original image nor the cropped region shows suspicious opacities or consolidation; both appear clear, so there is no evidence of pneumonia. </think>. My answer: \\boxed{no}.  "
                            "Your ultimate goal is to diagnosis after the valid tool use, the thinking process is to serve them, and you can answer after you think well."
                            "Your diagnosis after tool use can be either yes or no. Now start with your tool use first."
            },
            {"role": "user", "content": prompt_text}
        ],
        "images": [img_dict],
        "ability": "chest_xray",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split_name,
            "index": seed,
            "answer": answer,
            "need_tool_kwargs": True,
            "tools_kwargs": {
                "crop_image":{
                    "create_kwargs": {"_dummy": None}
                }
            },
            "disease": disease,
            "dicom_id": row["dicom_id"],
        },
    }


# "tools_kwargs": {
#     "calc_gsm8k_reward": {
#         "create_kwargs": {"ground_truth": solution},
#         # "execute_kwargs": {},
#         # "calc_reward_kwargs": {},
#         # "release_kwargs": {},
#     },
# },


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # local
    # parser.add_argument("--full_parquet", default="~/verl_new/cxr_mini_crop/full.parquet")
    # parser.add_argument("--train_parquet", default="~/verl_new/cxr_mini_crop/train.parquet")
    # parser.add_argument("--test_parquet", default="~/verl_new/cxr_mini_crop/test.parquet")
    # hpc

    # cp -r /mnt/input/ms_cxr_data/cxr_mini_crop/* ~/verl_new/cxr_mini_crop/

    parser.add_argument("--full_parquet", default="~/verl_new/cxr_data_process/ms_cxr_data/full.parquet")
    parser.add_argument("--train_parquet", default="~/verl_new/cxr_data_process/mimic_cxr_jpg_data/fold_data/fold0/fold0_train_subset_5000_parquet.parquet")
    parser.add_argument("--test_parquet", default="~/verl_new/cxr_data_process/ms_cxr_data/ms_cxr_all.parquet")
    parser.add_argument("--local_dir", default="~/data/cxr_5k_tool")
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
    pd.DataFrame(train_data).to_parquet(os.path.join(args.local_dir, "train_neg_prompt.parquet"))
    pd.DataFrame(test_data).to_parquet(os.path.join(args.local_dir, "test_neg_prompt.parquet"))

    # -------------------------
    # Optional HDFS upload
    # -------------------------
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)

    print("✅ Conversion finished. Train / test parquet saved to", args.local_dir)
