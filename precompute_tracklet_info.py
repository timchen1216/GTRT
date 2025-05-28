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
)
from TrackletData import TrackletData
from config import device, soft_label

# Path to save precomputed data
output_dir = "precomputed_data"
tracklet_data_save_path = os.path.join(output_dir, "tracklet_data.pt")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Function to precompute tracklet information
def precompute_tracklet_info():
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
        transform=det_transforms,
        stride=tracklet_temporal_stride,
    )
    gt_data = CreateMOTDataset(
        data_path=train_gt_path,
        temporal_len=tracklet_temporal_len,
        transform=gt_transforms,
        stride=tracklet_temporal_stride,
    )
    print("MOT data length:", len(mot_data))
    print("GT data length:", len(gt_data))

    dataloader = torch.utils.data.DataLoader(
        mot_data,
        batch_size=1,  # Process one sample at a time
        num_workers=4,
        collate_fn=None,
    )
    gt_dataloader = torch.utils.data.DataLoader(
        gt_data,
        batch_size=1,
        num_workers=4,
        collate_fn=None,
    )

    # Initialize storage for results with proper empty tensors
    merged_info = {
        "tracklet_embs": [],
        "tracklet_scores": [],
        "tracklet_labels": [],
        "A": [],
        "binary_label": [],
        "edge_idx": [],
        "tracklet_gt_embs": [],
    }

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

        tracklet_dict = {}
        # for k, v in batch.items():
        #     print(k)
        for k, v in batch.items():
            if k != "img_paths":
                tracklet_dict[k] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[k] = v
        for k, v in gt_batch.items():
            if k != "img_paths":
                tracklet_dict[f"gt_{k}"] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[f"gt_{k}"] = v

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
        tracklet_info = get_tracklet_info(
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

        for k, v in tracklet_info.items():
            merged_info[k].append(v)
        #     print(k, v.shape)
        # print()
        # if cnt == 160:
        #     break

    # Find maximum dimensions
    max_dims = {
        "tracklet_embs": [0, 4, 65],
        "tracklet_scores": [0, 1, 65],
        "tracklet_labels": [0],
        "A": [0, 0],
        "binary_label": [0, 0],
        "edge_idx": [0, 2],
        "tracklet_gt_embs": [0, 4, 65],
    }

    for k, batch_tensors in merged_info.items():
        for tensor in batch_tensors:
            shape = list(tensor.size())
            if len(shape) == 3:  # For 3D tensors
                max_dims[k][0] = max(max_dims[k][0], shape[0])
            elif len(shape) == 2:  # For 2D tensors
                if k == "edge_idx":
                    max_dims[k][0] = max(max_dims[k][0], shape[0])
                else:
                    max_dims[k][0] = max(max_dims[k][0], shape[0])
                    max_dims[k][1] = max(max_dims[k][1], shape[1])
            else:  # For 1D tensors
                max_dims[k][0] = max(max_dims[k][0], shape[0])
    # print("Maximum dimensions:")
    # for k, v in max_dims.items():
    #     print(k, v)

    # Pad and stack tensors
    for k, v in merged_info.items():
        padded_tensors = []
        for tensor in v:
            if k in ["tracklet_embs", "tracklet_scores", "tracklet_gt_embs"]:
                # Pad first dimension only
                padding_size = max_dims[k][0] - tensor.size(0)
                if padding_size > 0:
                    padding = torch.zeros(
                        padding_size, *tensor.size()[1:], device=tensor.device
                    )
                    tensor = torch.cat([tensor, padding], dim=0)
            elif k in ["A", "binary_label"]:
                # Pad both dimensions
                pad_rows = max_dims[k][0] - tensor.size(0)
                pad_cols = max_dims[k][1] - tensor.size(1)
                if pad_rows > 0 or pad_cols > 0:
                    padded = torch.zeros(
                        max_dims[k][0], max_dims[k][1], device=tensor.device
                    )
                    padded[: tensor.size(0), : tensor.size(1)] = tensor
                    tensor = padded
            elif k == "edge_idx":
                # Pad only rows if needed
                pad_rows = max_dims[k][0] - tensor.size(0)
                if pad_rows > 0:
                    padding = torch.zeros(pad_rows, 2, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=0)
            elif k == "tracklet_labels":  # Add handling for tracklet_labels
                # Pad the 1D tensor to match max dimension
                padding_size = max_dims[k][0] - tensor.size(0)
                if padding_size > 0:
                    padding = torch.zeros(padding_size, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=0)
            padded_tensors.append(tensor)

        # print(f"Stacking {k} with shape {padded_tensors[0].shape}")
        merged_info[k] = torch.stack(padded_tensors, dim=0)

    print(f"Saving tracklet information to {tracklet_data_save_path}")
    for k, v in merged_info.items():
        print(f"{k}: {v.shape}")
    # Save the merged information
    tracklet_data = TrackletData(merged_info)
    torch.save(tracklet_data, tracklet_data_save_path)

    # Verify the saved data
    load_data = torch.load(tracklet_data_save_path, map_location="cpu")
    print("Loaded data verification:", load_data)

    print("Precomputation completed successfully!")


if __name__ == "__main__":
    precompute_tracklet_info()
