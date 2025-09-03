import subprocess
import time
import os

# 源和目标路径
SRC = "/home/aiscuser/verl_new/checkpoints/cxr_8k_bal_tool_qwen2.5-vl-7b-instruct_7.0.6_dapo_second_try_full_test/"
DST = "/mnt/input/cxr_crop/ours_1think_1crop/cxr_8k_bal_tool_qwen2.5-vl-7b-instruct_7.0.6_dapo_second_try_full_test/"

# 确保目标目录存在
os.makedirs(DST, exist_ok=True)

while True:
    try:
        print("Running rsync...")
        subprocess.run(
            ["rsync", "-av", SRC, DST],
            check=True
        )
        print("Sync complete. Waiting 10 minutes...")
    except subprocess.CalledProcessError as e:
        print("Error during rsync:", e)

    # 等待600秒（10分钟）
    time.sleep(600)
