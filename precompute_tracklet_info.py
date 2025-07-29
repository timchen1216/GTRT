import os
import torch
from tqdm import tqdm
from mot_data_loader import (
    CreateMOTDataset,
    BoxClip,
    TrackletSplit,
    AddFP,
    RandomDelete,
)
from head_utils import get_tracklet_info
from torchvision import transforms
from config import (
    train_data_path,
    train_gt_path,
    val_data_path,
    val_gt_path,
    tracklet_temporal_len,
    tracklet_temporal_stride,
    T_tracklet_stride,
)
from TrackletData import TrackletData
from config import device, soft_label

# Path to save precomputed data
output_dir = "precomputed_data"
tracklet_data_save_path = os.path.join(output_dir, "tracklet_data.pt")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Function to precompute tracklet information
def precompute_train_info():
    print("Initializing precomputation...")

    # Data loader
    base_transforms = transforms.Compose([BoxClip()])

    det_transforms = transforms.Compose(
        [
            TrackletSplit(),
            AddFP(temporal_len=tracklet_temporal_len),
            RandomDelete(),
            BoxClip(),
        ]
    )
    gt_transforms = transforms.Compose([BoxClip()])

    mot_data = CreateMOTDataset(
        data_path=train_data_path,
        temporal_len=tracklet_temporal_len,
        transform=base_transforms,
        stride=T_tracklet_stride,
        random_skip=False,
    )
    gt_data = CreateMOTDataset(
        data_path=train_gt_path,
        temporal_len=tracklet_temporal_len,
        transform=gt_transforms,
        stride=T_tracklet_stride,
        random_skip=False,
    )
    print("MOT data length:", len(mot_data))
    print("GT data length:", len(gt_data))

    dataloader = torch.utils.data.DataLoader(
        mot_data,
        batch_size=1,  # Process one sample at a time
        num_workers=4,
        shuffle=False,  # 確保按順序取樣
        collate_fn=None,
    )

    gt_dataloader = torch.utils.data.DataLoader(
        gt_data,
        batch_size=1,
        num_workers=4,
        shuffle=False,  # 確保按順序取樣
        collate_fn=None,
    )

    # Initialize storage for results with proper empty tensors
    seq_info_list = []

    print("Processing sequences...")
    cnt = 0
    for batch, gt_batch in tqdm(
        zip(dataloader, gt_dataloader),
        desc="Processing with padding",
        total=len(dataloader),
    ):
        # cnt += 1
        # if cnt <= 145:
        #     continue
        # if cnt == 160:
        #     break

        tracklet_dict = {}
        # for k, v in batch.items():
        #     print(k)
        for k, v in batch.items():
            if k != "img_paths" and k != "video_name":
                tracklet_dict[k] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[k] = v
        for k, v in gt_batch.items():
            if k != "img_paths" and k != "video_name":
                tracklet_dict[f"gt_{k}"] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[f"gt_{k}"] = v
        # for k, v in tracklet_dict.items():
        #     print(k)
        tracklet_dict["boxes"][:, 0::2] = tracklet_dict["boxes"][:, 0::2] / float(
            batch["width"].item()
        )
        tracklet_dict["boxes"][:, 1::2] = tracklet_dict["boxes"][:, 1::2] / float(
            batch["height"].item()
        )
        tracklet_dict["gt_boxes"][:, 0::2] = tracklet_dict["gt_boxes"][:, 0::2] / float(
            gt_batch["width"].item()
        )
        tracklet_dict["gt_boxes"][:, 1::2] = tracklet_dict["gt_boxes"][:, 1::2] / float(
            gt_batch["height"].item()
        )
        # soft label
        if soft_label:
            score = 0.9 + 0.1 * torch.rand(
                tracklet_dict["boxes"].shape[0], device=device
            )
        else:
            score = torch.ones(tracklet_dict["boxes"].shape[0], device=device)
        window_info = get_tracklet_info(
            det_ids=tracklet_dict["obj_ids"],
            gt_ids=tracklet_dict["gt_obj_ids"],
            det_fr_ids=tracklet_dict["fr_ids"],
            gt_fr_ids=tracklet_dict["gt_fr_ids"],
            det_bboxes=tracklet_dict["boxes"],
            gt_bboxes=tracklet_dict["gt_boxes"],
            scores=score,  # 使用生成的 score
            temporal_len=tracklet_temporal_len,
            device=device,
            stage="train",
        )
        window_info["time_window"] = torch.tensor(
            [tracklet_dict["start_frame"].item(), tracklet_dict["end_frame"].item()],
            dtype=torch.int64,
            device=device,
        )
        window_info["seq_name"] = tracklet_dict["video_name"]
        # if window_info["time_window"][0] == 1:
        #     print(window_info["seq_name"])
        #     print(window_info["time_window"])
        # for k, v in window_info.items():
        #     print(k, v.shape if torch.is_tensor(v) else v)

        seq_info_list.append(window_info)
    # print(len(seq_info_list))
    for k, v in seq_info_list[-1].items():
        print(k, v.shape if torch.is_tensor(v) else v)

    print(f"Saving tracklet information to {tracklet_data_save_path}")

    torch.save(seq_info_list, tracklet_data_save_path)

    # Verify the saved data
    load_data = torch.load(tracklet_data_save_path, map_location="cpu")
    # print(type(load_data))
    # for k, v in load_data[-1].items():
    #     print(k, v.shape if torch.is_tensor(v) else v)

    # print("Loaded data verification:", load_data)

    print("Precomputation completed successfully!")


def precompute_val_info():
    print("Initializing validation data precomputation...")

    base_transforms = transforms.Compose([BoxClip()])

    val_data = CreateMOTDataset(
        data_path=val_data_path,
        temporal_len=tracklet_temporal_len,
        transform=base_transforms,
        stride=tracklet_temporal_stride,
        random_skip=False,
    )

    val_gt_data = CreateMOTDataset(
        data_path=val_gt_path,
        temporal_len=tracklet_temporal_len,
        transform=base_transforms,
        stride=tracklet_temporal_stride,
        random_skip=False,
    )

    print("Validation data length:", len(val_data))
    print("Validation GT data length:", len(val_gt_data))

    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        collate_fn=None,
    )

    val_gt_dataloader = torch.utils.data.DataLoader(
        val_gt_data,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        collate_fn=None,
    )
    seq_info_list = []

    print("Processing validation sequences...")
    cnt = 0
    for val_batch, val_gt_batch in tqdm(
        zip(val_dataloader, val_gt_dataloader),
        desc="Processing validation data",
        total=len(val_dataloader),
    ):
        # cnt += 1
        # if cnt <= 145:
        #     continue
        # if cnt == 160:
        #     break

        tracklet_dict = {}
        for k, v in val_batch.items():
            if k != "img_paths" and k != "video_name":
                tracklet_dict[k] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[k] = v

        for k, v in val_gt_batch.items():
            if k != "img_paths" and k != "video_name":
                tracklet_dict[f"gt_{k}"] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[f"gt_{k}"] = v

        # Normalize boxes
        tracklet_dict["boxes"][:, 0::2] = tracklet_dict["boxes"][:, 0::2] / float(
            val_batch["width"].item()
        )
        tracklet_dict["boxes"][:, 1::2] = tracklet_dict["boxes"][:, 1::2] / float(
            val_batch["height"].item()
        )
        tracklet_dict["gt_boxes"][:, 0::2] = tracklet_dict["gt_boxes"][:, 0::2] / float(
            val_gt_batch["width"].item()
        )
        tracklet_dict["gt_boxes"][:, 1::2] = tracklet_dict["gt_boxes"][:, 1::2] / float(
            val_gt_batch["height"].item()
        )

        score = torch.ones(
            tracklet_dict["boxes"].shape[0], device=device
        )  # 驗證集不使用soft label

        val_window_info = get_tracklet_info(
            det_ids=tracklet_dict["obj_ids"],
            gt_ids=tracklet_dict["gt_obj_ids"],
            det_fr_ids=tracklet_dict["fr_ids"],
            gt_fr_ids=tracklet_dict["gt_fr_ids"],
            det_bboxes=tracklet_dict["boxes"],
            gt_bboxes=tracklet_dict["gt_boxes"],
            scores=score,
            temporal_len=tracklet_temporal_len,
            device=device,
            stage="train",
        )

        # Add time window information for validation data
        val_window_info["time_window"] = torch.tensor(
            [tracklet_dict["start_frame"].item(), tracklet_dict["end_frame"].item()],
            dtype=torch.int64,
            device=device,
        )
        val_window_info["seq_name"] = tracklet_dict["video_name"]
        # if val_window_info["time_window"][0] == 1:
        #     print(val_window_info["seq_name"])
        #     print(val_window_info["time_window"])

        seq_info_list.append(val_window_info)

    for k, v in seq_info_list[-1].items():
        print(k, v.shape if torch.is_tensor(v) else v)

    # 處理和儲存驗證數據
    val_tracklet_data_save_path = os.path.join(output_dir, "val_tracklet_data.pt")
    print(f"Saving validation tracklet information to {val_tracklet_data_save_path}")

    torch.save(seq_info_list, val_tracklet_data_save_path)
    print("Validation data precomputation completed successfully!")


def precompute_tracklet_info():
    print("Starting precomputation pipeline...")

    # 原有的訓練數據預計算
    precompute_train_info()

    # 添加驗證數據預計算
    precompute_val_info()

    print("All precomputation completed!")


if __name__ == "__main__":
    precompute_tracklet_info()
