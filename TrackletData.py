import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TrackletData(nn.Module):
    def __init__(
        self,
        merged_info,
    ):
        super(TrackletData, self).__init__()
        self.tracklet_embs = merged_info["tracklet_embs"]
        self.tracklet_scores = merged_info["tracklet_scores"]
        self.tracklet_labels = merged_info["tracklet_labels"]
        self.A = merged_info["A"]
        self.binary_label = merged_info["binary_label"]
        self.edge_idx = merged_info["edge_idx"]
        self.tracklet_gt_embs = merged_info["tracklet_gt_embs"]
        self.time_window = merged_info["time_window"]
        # self.tracklet_gt_embs = merged_info["tracklet_gt_embs"]
        # self.tracklet_scores = []
        # self.tracklet_labels = []
        # self.A = []
        # self.binary_label = []
        # self.edge_idx = []
        # self.tracklet_gt_embs = []

        # # Collect all tensors first to find max sizes
        # for tracklet_info in merged_info:
        #     self.tracklet_embs.append(tracklet_info["tracklet_embs"])
        #     self.tracklet_scores.append(tracklet_info["tracklet_scores"])
        #     self.tracklet_labels.append(tracklet_info["tracklet_labels"])
        #     self.A.append(tracklet_info["A"])
        #     self.binary_label.append(tracklet_info["binary_label"])
        #     self.edge_idx.append(tracklet_info["edge_idx"])
        #     self.tracklet_gt_embs.append(tracklet_info["tracklet_gt_embs"])

        # # Find max sizes
        # def max_size(tensors, dim=0):
        #     return max(t.size(dim) for t in tensors)

        # # torch.tensor(self.tracklet_embs.shape).argmax().item()

        # # Pad tensors to max size
        # def pad_tensor(tensor, max_size):
        #     pad_size = max_size - tensor.size(0)

        #     if pad_size > 0:
        #         padding = torch.zeros(
        #             (pad_size,) + tensor.size()[1:],
        #             dtype=tensor.dtype,
        #             device=tensor.device,
        #         )
        #         return torch.cat([tensor, padding], dim=0)
        #     return tensor

        # tracklet_embs_max_size = max_size(self.tracklet_embs)
        # tracklet_scores_max_size = max_size(self.tracklet_scores)
        # tracklet_labels_max_size = max_size(self.tracklet_labels)
        # A_max_size = max_size(self.A, 1)
        # # binary_label_max_size = max_size(self.binary_label)
        # # edge_idx_max_size = max_size(self.edge_idx)
        # # tracklet_gt_embs_max_size = max_size(self.tracklet_gt_embs)

        # # Pad and stack tensors
        # self.tracklet_embs = torch.stack(
        #     [pad_tensor(t, tracklet_embs_max_size) for t in self.tracklet_embs], dim=0
        # )
        # self.tracklet_scores = torch.stack(
        #     [pad_tensor(t, tracklet_scores_max_size) for t in self.tracklet_scores],
        #     dim=0,
        # )
        # self.tracklet_labels = torch.stack(
        #     [pad_tensor(t, tracklet_labels_max_size) for t in self.tracklet_labels],
        #     dim=0,
        # )
        # self.A = torch.stack(self.A, dim=0)
        # # self.A = torch.stack([pad_tensor(t, A_max_size) for t in self.A], dim=0)
        # self.binary_label = torch.stack(
        #     self.binary_label, dim=0
        # )  # Assuming binary_label doesn't need padding
        # self.edge_idx = torch.stack(
        #     self.edge_idx, dim=0
        # )  # Assuming edge_idx doesn't need padding
        # self.tracklet_gt_embs = torch.stack(
        #     [pad_tensor(t, max_size) for t in self.tracklet_gt_embs], dim=0
        # )


class TrackletDataset(Dataset):
    def __init__(self, tracklet_data):
        super(TrackletDataset, self).__init__()
        # self.tracklet_data = tracklet_data
        self.tracklet_embs = tracklet_data.tracklet_embs
        self.tracklet_scores = tracklet_data.tracklet_scores
        self.tracklet_labels = tracklet_data.tracklet_labels
        self.A = tracklet_data.A
        self.binary_label = tracklet_data.binary_label
        self.edge_idx = tracklet_data.edge_idx
        # print("edge_idx", self.edge_idx.shape)
        self.tracklet_gt_embs = tracklet_data.tracklet_gt_embs
        self.time_window = tracklet_data.time_window

    def __len__(self):
        return len(self.tracklet_embs)

    def __getitem__(self, idx):
        return {
            "tracklet_embs": self.tracklet_embs[idx],
            "tracklet_scores": self.tracklet_scores[idx],
            "tracklet_labels": self.tracklet_labels[idx],
            "A": self.A[idx],
            "binary_label": self.binary_label[idx],
            "edge_idx": self.edge_idx[idx],
            "tracklet_gt_embs": self.tracklet_gt_embs[idx],
            "time_window": self.time_window[idx],
        }
