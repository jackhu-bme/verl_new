import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/home/aiscuser/data/cxr_mini/test-output-8.parquet')

# 转换为 CSV
df.to_csv('/home/aiscuser/data/cxr_mini/test-output-8.csv', index=False)
# 转换为 JSON
df.to_json('/home/aiscuser/data/cxr_mini/test-output-8.json', orient='records', lines=True)
