import os
import pandas as pd
import cv2
import multiprocessing
from tqdm import tqdm

# 读取数据
df = pd.read_parquet('/home/aiscuser/verl_new/cxr_data_process/ms_cxr_data/full_v2.parquet')

# 获取唯一的原始路径列表
unique_paths = df['img_original_path'].unique().tolist()

# 替换路径前缀
prefix = "/mnt/input/mimic_cxr_jpg/ori_dataset/mimic-cxr-jpg-2.1.0/mimic-cxr-jpg-2.1.0.physionet.org/files"
prefix_256 = "/mnt/input/mimic_cxr_jpg/ori_dataset/mimic-cxr-jpg-2.1.0/mimic-cxr-jpg-2.1.0.physionet.org/files_256"

def process_image_path(original_path):
    new_path = original_path.replace(prefix, prefix_256)

    # 创建目标文件夹
    new_dir = os.path.dirname(new_path)
    # if not os.path.exists(new_dir):
    os.makedirs(new_dir, exist_ok=True)

    # 检查原始文件是否存在
    if not os.path.exists(original_path):
        print(f"Warning: 原始文件不存在: {original_path}")
    
    # 读取图像并进行 resize
    img = cv2.imread(original_path)
    if img is not None:
        # Resize 图像为 256x256
        resized_img = cv2.resize(img, (256, 256))

        # 检查新文件是否已存在，如果存在，打印警告
        if os.path.exists(new_path):
            print(f"Warning: 新文件已存在并将被替换: {new_path}")
        
        # 保存 resized 图像
        cv2.imwrite(new_path, resized_img)
    
    return new_path  # 返回新路径以便在进程完成后追踪

def parallel_process(unique_paths, num_processes=192):
    # 使用 tqdm 显示进度条
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image_path, unique_paths), total=len(unique_paths)))
    return results

# 开始处理并显示进度
new_paths = parallel_process(unique_paths)

print("任务完成，所有图像已保存到新的路径！")
