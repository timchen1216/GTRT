dataset = "Dance"
sub_class = "pedestrian"
data_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack"

# Paths for ground truth data
train_gt_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/train_gt"
val_gt_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/val_gt"
train_seqmap_path = "/home/caig/data/GTRT_data/datasets/DanceTrack/train_seqmap.txt"
val_seqmap_path = "/home/caig/data/GTRT_data/datasets/DanceTrack/val_seqmap.txt"

# Paths for detection data
train_det_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/MOTIP_train"
val_det_dir = "/home/caig/data/GTRT_data/datasets/DanceTrack/MOTIP_val"

# Paths for annotations
gt_path = "TrackAnnos/GT_DanceTrack_train.json"
train_data_path = "TrackAnnos/MOTIP_DanceTrack_train.json"
test_data_path = "TrackAnnos/MOTIP_DanceTrack_test.json"

# Model paths
# det_graph_model_load_path = "models/local_model_KITTI.tar"
tracklet_graph_model_load_path = "models/tracklet_graph_model_epoch_10.tar"

# Output directories
save_tracklet_graph_img_dir = "save_img/DanceTrack_test"
save_txt_dir = "save_txt/DanceTrack_test"

# Flags for saving outputs
save_img = True
save_txt = True

# Device configuration
device = "cuda:0"

# Temporal and batch settings
det_temporal_len = 16
tracklet_temporal_len = 64
T_det_window = 17
T_det_stride = 5
T_tracklet_stride = 5
batch_size = 8

# Thresholds and post-processing
emb_dist_thresh = 0.7
tracklet_associate_thresh = 0.2
remove_N = 3
post_precessing = True
