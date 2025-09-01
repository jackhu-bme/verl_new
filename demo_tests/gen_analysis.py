import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

rollout_dir = "/home/aiscuser/verl_new/generations/cxr_5k_tool_qwen2.5-vl-7b-instruct_7.0.2_dapo_first_try/rollout"
name = "dapo"

results = []

# 找到所有符合 *.jsonl 的文件，并提取其中的数字 i
file_list = []
for fname in os.listdir(rollout_dir):
    if fname.endswith(".jsonl"):
        match = re.match(r"(\d+)\.jsonl", fname)
        if match:
            file_list.append(int(match.group(1)))

# 按 i 排序
file_list = sorted(file_list)

# 遍历所有文件
for i in file_list:
    file_path = os.path.join(rollout_dir, f"{i}.jsonl")
    yes_count, no_count = 0, 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            output = record.get("output", "")
            yes_count += output.count("{yes}")
            no_count += output.count("{no}")

    results.append((i, yes_count, no_count))

# 保存 DataFrame
df = pd.DataFrame(results, columns=["i", "yes_count", "no_count"])

csv_path = f"rollout_results_{name}.csv"
df.to_csv(csv_path, index=False)

# 画图
plt.figure(figsize=(8,5))
plt.plot(df["i"], df["yes_count"], label="yes count", marker="o")
plt.plot(df["i"], df["no_count"], label="no count", marker="x")
plt.xlabel("File index (i)")
plt.ylabel("Count")
plt.title(f"Rollout results ({name})")
plt.legend()
plt.grid(True)
plot_path = f"rollout_results_{name}.png"
plt.savefig(plot_path)
plt.close()

print(csv_path, plot_path)
