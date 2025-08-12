import pandas as pd

# 读取原始 Parquet 文件
df = pd.read_parquet('~/verl_new/cxr_mini_crop/full.parquet')

# 定义需要替换路径前缀的列名
path_columns = ['img_original_path', 'img_256_path', 'img_vis_path']

# 替换路径前缀
for col in path_columns:
    df[col] = df[col].str.replace('/mnt/input/ms_cxr_data', '/home/yqh/Qwen2.5-VL/datasets', regex=False)

# 保存为新的 Parquet 文件
df.to_parquet('~/verl_new/cxr_mini_crop/full_local.parquet', index=False)

print("路径替换完成，保存为 full_replaced.parquet")
