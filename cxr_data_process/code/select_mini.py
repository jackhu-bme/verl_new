import pandas as pd


def sample(input_path, output_path, n=20):
    """
    从输入的 parquet 文件中随机采样 20 行，保存为新的 parquet 文件。
    """
    # 读取 parquet
    df = pd.read_parquet(input_path)

    # 随机采样 20 行
    sampled_df = df.sample(n=n, random_state=42)

    # 保存成新的 parquet
    sampled_df.to_parquet(output_path, index=False)

    print(f"保存完成: {output_path}")

# 输入和输出路径
# input_path_train = "/home/aiscuser/data/cxr_5k_tool/train.parquet"
# output_path_train = "/home/aiscuser/data/cxr_5k_tool/train_mini.parquet"

# sample(input_path_train, output_path_train)

# test
input_path_test = "/home/aiscuser/data/cxr_20k_bal_tool_v2_2_think/test.parquet"
output_path_test = "/home/aiscuser/data/cxr_20k_bal_tool_v2_2_think/test_mini.parquet"

sample(input_path_test, output_path_test, n=100)