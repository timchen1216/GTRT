import numpy as np
import json
import os
from PIL import Image

import data_converter



# write data
root_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/test"
det_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/MOTIP_test"
seq_map = "/home/caig/data/GTRT_data/datasets/DanceTrack/test_seqmap.txt"
save_path = "TrackAnnos/MOTIP_test.json"


global_cnt = 0
track_data = {}
video_list = os.listdir(root_dir)
det_list = os.listdir(det_dir)
print(video_list)
for n in range(len(det_list)): 

    video_name = det_list[n]
    print(video_name)
    det_name = det_list[n][:-4]
    video_dir = root_dir+"/"+det_name

    det_path = det_dir+"/"+det_list[n]
    track_data, global_cnt = data_converter.convert_MOT(track_data, det_name, video_dir, det_path, None, "test", None, global_cnt)

track_data = data_converter.index_data(track_data)
print(global_cnt)

# write to json
with open(save_path, 'w') as outfile:
    json.dump(track_data, outfile)
