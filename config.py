import torch


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
val_det_dir = "/home/caig/data/GTRT_data/datasets/DanceTracks/MOTIP_val"

# Paths for annotations
train_data_path = "TrackAnnos/MOTIP_DanceTrack_train.json"
train_gt_path = "TrackAnnos/GT_DanceTrack_train.json"
val_data_path = "TrackAnnos/MOTIP_DanceTrack_val.json"
val_gt_path = "TrackAnnos/GT_DanceTrack_val.json"
test_data_path = "TrackAnnos/MOTIP_DanceTrack_test.json"

soft_label = False

# Model paths
# det_graph_model_load_path = "models/local_model_KITTI.tar"
# tracklet_graph_model_load_path = "models/tracklet_graph_model_best_DA.tar"
tracklet_graph_model_load_path = "weights/gtrt_tf.tar"

# Output directories
save_tracklet_graph_img_dir = "save_img/DanceTrack_test"
save_txt_dir = "save_txt/DanceTrack_test"

# Flags for saving outputs
save_img = True
save_txt = True

# Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Correctly define the device


# Temporal and batch settings
det_temporal_len = 16
tracklet_temporal_len = 32
T_det_window = 17
T_det_stride = 5
T_tracklet_stride = 4

# Model parameters
num_epoch = 10
num_id_vocabulary = 50
emb_dim = 256
hidden_dim = 512
num_layers = 6
num_heads = 8
dropout_prob = 0.1
max_length = 10000

# teacher_forcing_prob = 0.5
teacher_forcing_coef = 0.6

tracklet_temporal_stride = 4  # New parameter to control frame interval
batch_size = 64
prefetch_factor = 1
num_workers_per_gpu = 4

# Thresholds and post-processing
emb_dist_thresh = 0.7
tracklet_associate_thresh = 0.2
remove_N = 3
post_precessing = False
