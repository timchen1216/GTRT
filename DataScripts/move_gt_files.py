import os
import shutil

# 原始 GT 文件的根目錄
source_root = "/home/caig/data/GTRT_data/datasets/DanceTrack/val"
# 目標 GT 文件的根目錄
target_root = "/home/caig/data/GTRT_data/datasets/DanceTrack/val_gt"
# 序列映射文件路徑
seq_map_path = "/home/caig/data/GTRT_data/datasets/DanceTrack/val_seqmap.txt"

# 確保目標目錄存在
os.makedirs(target_root, exist_ok=True)

# 從 seq_map 讀取序列名稱
def read_seq_map(seq_map_path):
    with open(seq_map_path, 'r') as f:
        lines = f.readlines()
    # 跳過標題行
    return [line.strip() for line in lines[1:] if line.strip()]

# 讀取序列名稱
sequences = read_seq_map(seq_map_path)

# 遍歷序列名稱
for seq_name in sequences:
    seq_path = os.path.join(source_root, seq_name, "gt", "gt.txt")
    if os.path.exists(seq_path):
        target_path = os.path.join(target_root, f"{seq_name}.txt")
        # 複製文件
        shutil.copy(seq_path, target_path)
        print(f"Copied {seq_path} to {target_path}")
    else:
        print(f"GT file not found for sequence: {seq_name}")
