import einops
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm import tqdm
import json
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from mot_data_loader import CreateMOTDataset, BoxClip
from build_det_graph import build_adj_graph
from models.gtrt import GTRT
from config import *
from head_utils import get_tracklet_info
from TrackletData import TrackletData, TrackletDataset


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors in tracklet data.
    Pads tensors to the maximum size in each dimension within the batch using -1 as padding value.
    """
    if len(batch) == 0:
        return {}

    # Get all keys from the first sample
    keys = batch[0].keys()
    collated_batch = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # Handle tensor data
        if isinstance(values[0], torch.Tensor):
            # Get the maximum dimensions for this key across all samples
            if len(values[0].shape) >= 2:
                # For multi-dimensional tensors, pad to max size in each dimension
                shapes = [v.shape for v in values]
                max_dims = [max(dim_sizes) for dim_sizes in zip(*shapes)]

                # Pad each tensor to the maximum dimensions
                padded_values = []
                for v in values:
                    # Calculate padding needed for each dimension (from the end)
                    padding = []
                    for i in range(len(v.shape) - 1, -1, -1):
                        pad_size = max_dims[i] - v.shape[i]
                        padding.extend([0, pad_size])

                    # Apply padding with -1 as padding value
                    if any(p > 0 for p in padding):
                        padded_v = torch.nn.functional.pad(v, padding, value=-1)
                    else:
                        padded_v = v
                    padded_values.append(padded_v)

                # Stack the padded tensors
                collated_batch[key] = torch.stack(padded_values, dim=0)
            else:
                # For 1D tensors, use pad_sequence with -1 padding
                collated_batch[key] = pad_sequence(
                    values, batch_first=True, padding_value=-1
                )

        # Handle non-tensor data (like lists, scalars, etc.)
        else:
            collated_batch[key] = values

    return collated_batch


def create_padding_mask(original_tensor, padded_tensor):
    """
    Create a mask indicating which elements are padding (True) and which are real data (False).
    """
    mask = torch.zeros_like(padded_tensor, dtype=torch.bool)
    for i, orig_shape in enumerate([t.shape for t in original_tensor]):
        # Mark padded areas as True
        slices = tuple(slice(0, dim) for dim in orig_shape)
        mask[i][slices] = False  # Real data
        # Everything else remains True (padding)
    return mask


# Function to train the tracklet graph model
def train_tracklet_graph_model(
    tracklet_graph_model, tracklet_data, optimizer, loss_weight, device
):
    tracklet_graph_model.train()
    total_loss = 0.0
    batch_loss_rec = 0.0
    batch_loss_trip = 0.0
    batch_loss_BCE = 0.0

    weight_rec, weight_trip, weight_BCE = loss_weight
    external_state = {
        "last_pred_labels": None,
        "last_pred_masks": None,
    }

    # 使用tqdm的正確方式來顯示進度
    progress_bar = tqdm(
        enumerate(tracklet_data),
        total=len(tracklet_data),
        desc="Training Tracklet Graph  ",
        leave=True,
        mininterval=0.1,  # 更新頻率設為0.1秒
    )

    for batch_idx, batch in progress_bar:
        batch_start = torch.cuda.Event(enable_timing=True)
        batch_end = torch.cuda.Event(enable_timing=True)

        batch_start.record()

        # Move batch to device all at once
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape} on {v.device}")
        #     else:
        #         print(f"{k}: {type(v)}")
        # print(batch["tracklet_labels"])
        # Forward pass
        # seq_info = prepare_seq_info(data=batch, device=device)
        batch["external_last_pred"] = external_state["last_pred_labels"]
        batch["external_last_masks"] = external_state["last_pred_masks"]
        result = tracklet_graph_model(data=batch, stage="train")
        if "pred_labels" in result and "pred_masks" in result:
            external_state["last_pred_labels"] = result["pred_labels"]
            external_state["last_pred_masks"] = result["pred_masks"]

        loss_BCE = result["bce_loss"]
        # print("loss_BCE:", loss_BCE.shape)
        # print(f"loss_BCE: {loss_BCE}")
        # loss = sum([l.mean() for l in loss_BCE])
        loss = loss_BCE.mean()  # Assuming loss_BCE is a list of losses
        # print(f"loss: {loss}")

        # 在這裡清零梯度
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()

        batch_end.record()
        torch.cuda.synchronize()
        batch_time = batch_start.elapsed_time(batch_end)

        # Update statistics
        total_loss += loss.item()
        # batch_loss_rec += sum([l.mean().item() for l in loss_rec])
        # batch_loss_trip += sum([l.mean().item() for l in loss_trip])
        batch_loss_BCE += sum([l.mean().item() for l in loss_BCE])
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Update progress bar with actual timing
        progress_bar.set_postfix(
            {
                "batch_time": f"{batch_time:.1f}ms",
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/ (batch_idx + 1):.4f}",
            }
        )
        # 立即刪除不需要的變量
        del batch, result, loss_BCE, loss
    # Return total loss and individual losses
    return (
        total_loss / len(tracklet_data),
        batch_loss_rec / len(tracklet_data),
        batch_loss_trip / len(tracklet_data),
        batch_loss_BCE / len(tracklet_data),
    )


# Function to validate the tracklet graph model
def validate_tracklet_graph_model(tracklet_graph_model, val_data, loss_weight, device):
    tracklet_graph_model.eval()
    total_val_loss = 0.0
    val_loss_rec = 0.0
    val_loss_trip = 0.0
    val_loss_BCE = 0.0

    weight_rec, weight_trip, weight_BCE = loss_weight
    external_state = {
        "last_pred_labels": None,
        "last_pred_masks": None,
    }

    progress_bar = tqdm(
        enumerate(val_data),
        total=len(val_data),
        desc="Validating Tracklet Graph",
        leave=True,
        mininterval=0.1,
    )

    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # loss_rec, loss_trip, loss_BCE = tracklet_graph_model(
            #     data=batch, device=device, stage="train"
            # )
            batch["external_last_pred"] = external_state["last_pred_labels"]
            batch["external_last_masks"] = external_state["last_pred_masks"]
            result = tracklet_graph_model(data=batch, stage="val")
            if "pred_labels" in result and "pred_masks" in result:
                external_state["last_pred_labels"] = result["pred_labels"]
                external_state["last_pred_masks"] = result["pred_masks"]
            loss_BCE = result["bce_loss"]
            batch_loss = loss_BCE.mean()

            # Update statistics
            total_val_loss += batch_loss.item()
            # val_loss_rec += sum([l.mean().item() for l in loss_rec])
            # val_loss_trip += sum([l.mean().item() for l in loss_trip])
            val_loss_BCE += sum([l.mean().item() for l in loss_BCE])

            progress_bar.set_postfix(
                {
                    "val_loss": f"{batch_loss.item():.4f}",
                    "avg_val_loss": f"{total_val_loss / (batch_idx + 1):.4f}",
                }
            )
            # 立即刪除不需要的變量
            del batch, result, loss_BCE, batch_loss

    return (
        total_val_loss / len(val_data),
        val_loss_rec / len(val_data),
        val_loss_trip / len(val_data),
        val_loss_BCE / len(val_data),
    )


# Main training function
def main():
    # Set up device and check GPU availability
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs!")
        device = torch.device("cuda")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize tracklet graph model
    tracklet_graph_model = GTRT(
        temporal_length=tracklet_temporal_len + 1,
        num_id_vocabulary=num_id_vocabulary,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        max_length=max_length,
    )

    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        tracklet_graph_model = nn.DataParallel(tracklet_graph_model)
    tracklet_graph_model.to(device)
    print("Tracklet graph model initialized.")

    # Optimizer
    tracklet_optimizer = optim.AdamW(tracklet_graph_model.parameters(), lr=1e-4)

    # Load precomputed tracklet data
    tracklet_data_path = "precomputed_data/dance_motip_tracklet_data_T64_S4.pt"
    val_data_path = "precomputed_data/dance_motip_val_tracklet_data_T64_S4.pt"
    # tracklet_data_path = "precomputed_data/tracklet_data.pt"
    # val_data_path = "precomputed_data/val_tracklet_data.pt"
    print(f"Loading tracklet data from {tracklet_data_path}...")
    print(f"Loading tracklet data from {val_data_path}...")
    data = torch.load(tracklet_data_path, map_location="cpu")
    val_data = torch.load(val_data_path, map_location="cpu")
    # dataset = TrackletDataset(data)
    # val_dataset = TrackletDataset(val_data)
    # print(f"Dataset loaded with attributes:", dataset.__dict__.keys())

    # if not dataset:
    #     raise ValueError("No valid data found in the dataset")
    seq_info = prepare_seq_info(data, device="cpu")
    val_seq_info = prepare_seq_info(val_data, device="cpu")

    # Create DataLoader with batch size 4 and custom collate function
    # tracklet_data = DataLoader(
    #     seq_info,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=min(8, 4 * torch.cuda.device_count()),  # 限制worker數量
    #     pin_memory=True,
    #     persistent_workers=True,  # 保持worker進程存活
    #     prefetch_factor=1,  # 預加載1個batch
    #     drop_last=True,
    #     collate_fn=custom_collate_fn,  # Use custom collate function
    # )

    tracklet_data = DataLoader(
        seq_info,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 必須為 0 以保持狀態
        pin_memory=False,  # num_workers=0 時設為 False
        persistent_workers=False,  # num_workers=0 時必須為 False
        prefetch_factor=None,  # num_workers=0 時設為 None
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    # val_data = DataLoader(
    #     val_seq_info,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=min(8, 4 * torch.cuda.device_count()),  # 限制worker數量
    #     pin_memory=True,
    #     persistent_workers=True,  # 保持worker進程存活
    #     prefetch_factor=1,  # 預加載1個batch
    #     drop_last=True,
    #     collate_fn=custom_collate_fn,  # Use custom collate function
    # )
    val_data = DataLoader(
        val_seq_info,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 限制worker數量
        pin_memory=False,
        persistent_workers=False,  # 保持worker進程存活
        prefetch_factor=None,  # 預加載1個batch
        drop_last=True,
        collate_fn=custom_collate_fn,  # Use custom collate function
    )

    print("Starting training...")
    num_epochs = 20
    best_val_loss = float("inf")
    loss_weight = [1.0, 1.0, 1.0]  # rec, triplet, BCE

    for epoch in range(num_epochs):
        try:
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train phase
            train_loss, train_loss_rec, train_loss_trip, train_loss_BCE = (
                train_tracklet_graph_model(
                    tracklet_graph_model,
                    tracklet_data,
                    tracklet_optimizer,
                    loss_weight,
                    device,
                )
            )
            print(
                f"Training   - Loss Breakdown - Reconstruction Loss: {train_loss_rec:.6f}, "
                f"Triplet Loss: {train_loss_trip:.6f}, BCE Loss: {train_loss_BCE:.6f}"
            )
            print(f"Training   - Total Loss: {train_loss:.6f}")

            # Validation phase
            val_loss, val_loss_rec, val_loss_trip, val_loss_BCE = (
                validate_tracklet_graph_model(
                    tracklet_graph_model, val_data, loss_weight, device
                )
            )
            print(
                f"Validation - Loss Breakdown - Reconstruction Loss: {val_loss_rec:.6f}, "
                f"Triplet Loss: {val_loss_trip:.6f}, BCE Loss: {val_loss_BCE:.6f}"
            )
            print(f"Validation - Total Loss: {val_loss:.6f}")

            # Save model if it has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = "weights/tracklet_graph_model_best.tar"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": tracklet_graph_model.state_dict(),
                        "optimizer_state_dict": tracklet_optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    save_path,
                )
                print("*" * 50)
                print(f"Saved best model with validation loss: {val_loss:.6f}")
                print("*" * 50)

            # Save regular checkpoint
            save_path = f"weights/tracklet_graph_model_epoch_{epoch + 1}.tar"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": tracklet_graph_model.state_dict(),
                    "optimizer_state_dict": tracklet_optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_path,
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("Out of memory error, cleaning up and continuing...")
                continue
            else:
                raise e


def prepare_seq_info(data, device):
    """
    Prepare sequence information from the data dictionary.
    This function extracts and rearranges the necessary tensors for the model.
    """
    # print("data shape:", len(data))
    # print(list(data[0]["tracklet_labels"].values()))

    seq_info = []
    id_map = torch.randperm(num_id_vocabulary) + 1
    # print("id map", id_map)
    for i in range(1, len(data)):
        N, D, T = data[i]["tracklet_bbox"].shape
        _N, _, _ = data[i - 1]["tracklet_bbox"].shape
        N_pad = max(N, _N)

        trajectory_bbox = -torch.ones((N_pad, D, T), dtype=torch.float32, device=device)
        trajectory_masks = torch.zeros((N_pad, D, T), dtype=torch.bool, device=device)
        trajectory_id_labels = -torch.ones((N_pad), dtype=torch.int64, device=device)
        history_bbox = -torch.ones((N_pad, D, T), dtype=torch.float32, device=device)
        history_masks = torch.zeros((N_pad, D, T), dtype=torch.bool, device=device)
        history_id_labels = -torch.ones((N_pad), dtype=torch.int64, device=device)

        # print(data[i]["tracklet_mask"].shape)

        trajectory_bbox[:N, :, :] = data[i]["tracklet_bbox"]
        trajectory_masks[:N, :, :] = data[i]["tracklet_mask"]
        trajectory_id_labels[:N] = data[i]["gt_labels"]
        trajectory_id_masks = trajectory_id_labels != -1
        history_bbox[:_N, :, :] = data[i - 1]["tracklet_bbox"]
        history_masks[:_N, :, :] = data[i - 1]["tracklet_mask"]
        history_id_labels[:_N] = data[i - 1]["gt_labels"]
        history_id_masks = history_id_labels != -1

        (
            trajectory_bbox,
            trajectory_masks,
            trajectory_id_labels,
            trajectory_id_masks,
            history_bbox,
            history_masks,
            history_id_labels,
            history_id_masks,
        ) = trajectory_augmentation(
            trajectory_bbox,
            trajectory_masks,
            trajectory_id_labels,
            trajectory_id_masks,
            history_bbox,
            history_masks,
            history_id_labels,
            history_id_masks,
            id_map,
        )
        # print(history_id_labels.shape)
        # print(history_id_masks.shape)

        unique_track_ids = torch.unique(trajectory_id_labels[trajectory_id_masks])
        unique_history_ids = torch.unique(history_id_labels[history_id_masks])
        newborn_track_ids = unique_track_ids[
            ~torch.isin(unique_track_ids, unique_history_ids)
        ]
        if len(newborn_track_ids) > 0:
            # print(data[i]["tracklet_labels"])
            for newborn_id in newborn_track_ids:
                trajectory_id_labels[trajectory_id_labels == newborn_id] = (
                    num_id_vocabulary + 1
                )
            # print(f"Trajectory ID labels: {trajectory_id_labels}")
            # print(f"History ID labels: {history_id_labels}")
            # print(f"unique track IDs: {unique_track_ids}")
            # print(f"unique history IDs: {unique_history_ids}")
            # print(f"Newborn track IDs: {newborn_track_ids}")
            # print(
            #     f"Newborn track ID {newborn_id} found, assigning to {num_id_vocabulary + 1}"
            # )

        window_info = {
            "tracklet_bbox": trajectory_bbox.to(device),
            "tracklet_mask": trajectory_masks.to(device),
            "tracklet_labels": trajectory_id_labels.to(device),
            "tracklet_id_mask": trajectory_id_masks.to(device),
            "history_bbox": history_bbox.to(device),
            "history_mask": history_masks.to(device),
            "history_labels": history_id_labels.to(device),
            "history_id_mask": history_id_masks.to(device),
            "gt_bboxes": data[i]["gt_bboxes"].to(device),
            "gt_mask": data[i]["gt_mask"].to(device),
            "binary_label": data[i]["binary_label"].to(device),
            "time_window": data[i]["time_window"].to(device),
            "history_time_window": data[i - 1]["time_window"].to(device),
        }
        seq_info.append(window_info)
        del trajectory_bbox, trajectory_id_labels, trajectory_masks
        del history_bbox, history_id_labels, history_masks
    return seq_info


def trajectory_augmentation(
    trajectory_bbox,
    trajectory_masks,
    trajectory_id_labels,
    trajectory_id_masks,
    history_bbox,
    history_masks,
    history_id_labels,
    history_id_masks,
    id_map,
):
    # print("Before", trajectory_id_labels)
    trajectory_id_labels = id_map[trajectory_id_labels]
    history_id_labels = id_map[history_id_labels]
    # print("After", trajectory_id_labels)

    # N, D, T = trajectory_bbox.shape
    # shuffle_indices = torch.randperm(N)
    # trajectory_bbox = trajectory_bbox[shuffle_indices]
    # trajectory_masks = trajectory_masks[shuffle_indices]
    # trajectory_id_labels = trajectory_id_labels[shuffle_indices]
    # trajectory_id_masks = trajectory_id_masks[shuffle_indices]

    # print("Before", history_id_labels)
    # print("Before", history_id_masks)

    # _N, _D, _T = history_bbox.shape
    # history_shuffle_indices = torch.randperm(_N)
    # history_bbox = history_bbox[history_shuffle_indices]
    # history_masks = history_masks[history_shuffle_indices]
    # history_id_labels = history_id_labels[history_shuffle_indices]
    # history_id_masks = history_id_masks[history_shuffle_indices]

    # print("After", history_id_labels)
    # print("After", history_id_masks)

    return (
        trajectory_bbox,
        trajectory_masks,
        trajectory_id_labels,
        trajectory_id_masks,
        history_bbox,
        history_masks,
        history_id_labels,
        history_id_masks,
    )


if __name__ == "__main__":
    main()
