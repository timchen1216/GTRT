import numpy as np
import json
import os
from PIL import Image

import data_converter

# 設定路徑
root_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/test"
seq_map_path = "/home/caig/data/GTRT_data/datasets/DanceTrack/test_seqmap.txt"

det_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/MOTIP_test"
det_save_path = "TrackAnnos/MOTIP_DanceTrack_test.json"


# 從 seq_map 讀取序列名稱
def read_seq_map(seq_map_path):
    with open(seq_map_path, "r") as f:
        lines = f.readlines()
    # 跳過標題行
    return [line.strip() for line in lines[1:] if line.strip()]


global_cnt = 0
det_track_data = {}

# 從 seq_map 讀取需要處理的序列
sequences = read_seq_map(seq_map_path)
print(f"Found {len(sequences)} sequences to process.")

for seq_name in sequences:
    print(f"Processing sequence: {seq_name}")

    # 構建正確的路徑
    video_dir = os.path.join(
        root_dir, seq_name, "img1"
    )  # DanceTrack 圖像存儲在 img1 子目錄中
    det_path = os.path.join(det_dir, f"{seq_name}.txt")

    # 檢查路徑是否存在
    if not os.path.exists(video_dir):
        print(f"Warning: Video directory {video_dir} does not exist. Skipping.")
        continue

    if not os.path.exists(det_path):
        print(f"Warning: Detection file {det_path} does not exist. Skipping.")
        continue

    # 轉換數據
    det_track_data, global_cnt = data_converter.convert_MOT(
        det_track_data, seq_name, video_dir, det_path, None, "test", None, global_cnt
    )

# 索引數據
det_track_data = data_converter.index_data(det_track_data)
print(f"Total annotations: {global_cnt}")

# 寫入 JSON 文件
print(f"Saving results to {det_save_path}")
with open(det_save_path, "w") as outfile:
    json.dump(det_track_data, outfile)

print("Conversion completed successfully!")
