import numpy as np
import torch
import torch.nn.functional as F
import scipy.spatial.distance as sci_dist
from scipy import interpolate
from collections import Counter
from config import soft_label


def euclidean_dist(x, y):

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)  # Updated to use the recommended signature
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining_v2(dist_mat, is_pos, is_neg):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    dist_ap, relative_p_inds = torch.max(dist_mat * is_pos, 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
        dist_mat * is_neg + (1 - is_neg) * 1000, 1, keepdim=True
    )

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def hard_example_mining(dist_mat, labels, return_inds=False):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # random sampling positive
    dist_ap, relative_p_inds = torch.max(dist_mat - is_neg * 1000, 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat + is_pos * 1000, 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:

        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0)
            .expand(N, N)
        )

        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data
        )
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data
        )

        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def get_triplet_loss(features, labels, margin=0.2):

    labels_tensor = labels
    dist_mat = euclidean_dist(features, features)
    dist_ap, dist_an = hard_example_mining(dist_mat, labels_tensor)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
    loss = ranking_loss(dist_an, dist_ap, y)
    return loss


def get_triplet_loss_v2(features, edge_idx, is_pos, is_neg, device, margin=0.2):
    N = len(features)
    dist_mat = 10 * torch.ones(N, N, dtype=torch.float32, device=device)
    dist_mat[edge_idx[0, :], edge_idx[1, :]] = torch.norm(
        features[edge_idx[0, :]].cpu() - features[edge_idx[1, :]].cpu(), dim=1
    ).to(device)
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()

    dist_ap, dist_an = hard_example_mining_v2(dist_mat, is_pos, is_neg)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
    loss = ranking_loss(dist_an, dist_ap, y)
    return loss


def get_BCE_loss(features, labels):
    labels_tensor = torch.from_numpy(np.array(labels)).float().to("cuda")
    labels_tensor1 = torch.unsqueeze(labels_tensor, 1)
    labels_tensor2 = torch.unsqueeze(labels_tensor, 0)
    binary_label = labels_tensor1 - labels_tensor2
    binary_label[binary_label != 0] = 1
    binary_label = torch.clamp(1 - binary_label, min=0.0, max=1.0)
    sim_mat = torch.clamp(
        (torch.mm(features, features.permute(1, 0)) + 1.0) / 2.0, min=0.0, max=1.0
    )
    loss = F.binary_cross_entropy(sim_mat, binary_label)
    return loss


def create_binary_label(labels_tensor):
    labels_tensor1 = torch.unsqueeze(labels_tensor, 1)
    labels_tensor2 = torch.unsqueeze(labels_tensor, 0)
    binary_label = labels_tensor1 - labels_tensor2
    binary_label[binary_label != 0] = 1
    binary_label = torch.clamp(1 - binary_label, min=0.0, max=1.0)
    return binary_label


def eval_association(
    graph_emb,
    edge_idx,
    fr_ids,
    det_scores=None,
    obj_ids=None,
    dist_thresh=None,
    binary_label=None,
):

    # testing
    if dist_thresh is not None:
        pred_label = torch.zeros(len(fr_ids), len(fr_ids), device="cuda")
        tmp_idx = torch.nonzero(edge_idx[0, :] < edge_idx[1, :])
        selected_edge_idx = edge_idx[:, tmp_idx[:, 0]]
        dist = torch.norm(
            graph_emb[selected_edge_idx[0, :], :]
            - graph_emb[selected_edge_idx[1, :], :],
            dim=1,
        )

        while True:
            min_dist, min_idx = torch.min(dist, 0)
            if min_dist > dist_thresh:
                break
            tmp_edge_idx = selected_edge_idx[:, min_idx]
            pred_label[tmp_edge_idx[0], tmp_edge_idx[1]] = 1
            pred_label[tmp_edge_idx[1], tmp_edge_idx[0]] = 1
            tmp_idx = torch.nonzero(selected_edge_idx[0, :] == tmp_edge_idx[0])[:, 0]
            dist[tmp_idx] = 100
        return pred_label

    num_bins = 30

    # remove redundant nodes for eval
    gt_label = binary_label.clone()
    gt_mask = torch.ones_like(gt_label, device="cuda")
    uniq_fr_ids = torch.unique(fr_ids)
    for n in range(len(uniq_fr_ids)):
        tmp_idx = torch.nonzero(fr_ids == uniq_fr_ids[n])
        tmp_obj_ids = obj_ids[tmp_idx[:, 0]]
        uniq_obj_ids = torch.unique(tmp_obj_ids)
        for k in range(len(uniq_obj_ids)):
            tmp_idx = torch.nonzero(
                (fr_ids == uniq_fr_ids[n]) * (obj_ids == uniq_obj_ids[k])
            )
            _, max_idx = torch.max(det_scores[tmp_idx[:, 0]], 0)

            tmp_clone1 = gt_label[tmp_idx[max_idx, 0], :].clone()
            tmp_clone2 = gt_label[:, tmp_idx[max_idx, 0]].clone()
            tmp_clone3 = gt_mask[tmp_idx[max_idx, 0], :].clone()
            tmp_clone4 = gt_mask[:, tmp_idx[max_idx, 0]].clone()
            gt_label[tmp_idx[:, 0], :] = 0.0
            gt_label[:, tmp_idx[:, 0]] = 0.0
            gt_mask[tmp_idx[:, 0], :] = 0.0
            gt_mask[:, tmp_idx[:, 0]] = 0.0
            gt_label[tmp_idx[max_idx, 0], :] = tmp_clone1
            gt_label[:, tmp_idx[max_idx, 0]] = tmp_clone2
            gt_mask[tmp_idx[max_idx, 0], :] = tmp_clone3
            gt_mask[:, tmp_idx[max_idx, 0]] = tmp_clone4

    F_metric = torch.zeros(num_bins, device="cuda")
    pred_label = torch.zeros(
        binary_label.shape[0], binary_label.shape[1], num_bins, device="cuda"
    )
    dist_thresh = torch.linspace(0.05, 1.5, num_bins, device="cuda")
    tmp_idx = torch.nonzero(edge_idx[0, :] < edge_idx[1, :])
    selected_edge_idx = edge_idx[:, tmp_idx[:, 0]]
    dist = torch.norm(
        graph_emb[selected_edge_idx[0, :], :] - graph_emb[selected_edge_idx[1, :], :],
        dim=1,
    )

    while True:
        min_dist, min_idx = torch.min(dist, 0)
        if min_dist > dist_thresh[-1]:
            break
        tmp_edge_idx = selected_edge_idx[:, min_idx]

        tmp_thresh = torch.nonzero(min_dist < dist_thresh)[:, 0]
        pred_label[tmp_edge_idx[0], tmp_edge_idx[1], tmp_thresh] = 1
        pred_label[tmp_edge_idx[1], tmp_edge_idx[0], tmp_thresh] = 1
        tmp_idx = torch.nonzero(selected_edge_idx[0, :] == tmp_edge_idx[0])[:, 0]
        dist[tmp_idx] = 100

    F1_metric = torch.zeros(num_bins, device="cuda")
    for n in range(num_bins):
        TP = torch.sum((pred_label[:, :, n] * gt_mask == 1) * (gt_label == 1))
        FP_FN = torch.sum(
            (pred_label[:, :, n] * gt_mask == 1) * (gt_label == 0)
        ) + torch.sum((pred_label[:, :, n] * gt_mask == 0) * (gt_label == 1))
        F1_metric[n] = 2.0 * TP / (2.0 * TP + FP_FN)

    return F1_metric


def get_pred_edge_label(N_node, dist, edge_idx, dist_thresh, device):
    pred_edge_label = torch.zeros(len(edge_idx), device=device)
    while True:

        min_dist, min_idx = torch.min(dist, 0)
        if min_dist > dist_thresh:
            break
        pred_edge_label[min_idx] = 1
        tmp_edge_idx = edge_idx[min_idx, :]
        tmp_idx = torch.nonzero(edge_idx[:, 0] == tmp_edge_idx[0])[:, 0]
        dist[tmp_idx] = 100
        tmp_idx = torch.nonzero(edge_idx[:, 1] == tmp_edge_idx[1])[:, 0]
        dist[tmp_idx] = 100

    return pred_edge_label


def get_tracklet_label(N_node, pred_edge_label, edge_idx):
    tracklet_label = -np.ones((N_node))
    cnt = 0
    for n in range(len(edge_idx)):
        if pred_edge_label[n] == 1:
            if tracklet_label[edge_idx[n, 0]] == -1:
                tracklet_label[edge_idx[n, 0]] = cnt
                tracklet_label[edge_idx[n, 1]] = cnt
                cnt += 1
            else:
                tracklet_label[edge_idx[n, 1]] = tracklet_label[edge_idx[n, 0]]

    for n in range(len(tracklet_label)):
        if tracklet_label[n] == -1:
            tracklet_label[n] = cnt
            cnt += 1

    return tracklet_label


def calculate_iou(det_bbox, gt_bboxes):
    """
    計算檢測邊界框與多個 GT 邊界框的 IoU。

    Args:
        det_bbox (Tensor): 檢測邊界框，形狀為 (4,)。
        gt_bboxes (Tensor): GT 邊界框，形狀為 (N, 4)。

    Returns:
        Tensor: IoU 值，形狀為 (N,)。
    """
    # print("det_bbox shape:", det_bbox.shape)
    # print("gt_bboxes shape:", gt_bboxes.shape)
    # print("det_bbox:", det_bbox)
    # print("gt_bboxes:", gt_bboxes)
    x1 = torch.max(det_bbox[0], gt_bboxes[:, 0])
    y1 = torch.max(det_bbox[1], gt_bboxes[:, 1])
    x2 = torch.min(det_bbox[2], gt_bboxes[:, 2])
    y2 = torch.min(det_bbox[3], gt_bboxes[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
    gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    union_area = det_area + gt_areas - inter_area

    return inter_area / union_area


def get_tracklet_info(
    det_ids,
    gt_ids,
    det_fr_ids,
    gt_fr_ids,
    det_bboxes,
    gt_bboxes,
    scores,
    temporal_len,
    device,
    stage="train",
):
    # Ensure tracklet_label is a PyTorch tensor
    # print("device:", device)
    if isinstance(det_ids, np.ndarray):
        det_ids = torch.from_numpy(det_ids).to(device)

    # Move tracklet_label to CPU before using np.unique
    uniq_tracklet_label = np.unique(det_ids.cpu().numpy())
    if uniq_tracklet_label[0] == -1:
        uniq_tracklet_label = uniq_tracklet_label[1:]

    N_tracklet = len(uniq_tracklet_label)

    tracklet_embs = torch.zeros(
        N_tracklet, det_bboxes.shape[1], temporal_len + 1, device=device
    )
    tracklet_scores = torch.zeros(N_tracklet, 1, temporal_len + 1, device=device)
    if stage == "train":
        tracklet_gt_bboxes = torch.zeros(
            N_tracklet, det_bboxes.shape[1], temporal_len + 1, device=device
        )
        gt_tracklet_labels = np.zeros((N_tracklet))

    for n in range(N_tracklet):

        # Get indices for the current tracklet
        tracklet_indices = det_ids == uniq_tracklet_label[n]
        if not tracklet_indices.any():
            continue  # Skip if no indices match

        # Normalize frame indices to ensure they are within bounds
        valid_fr_ids = det_fr_ids[tracklet_indices].int().tolist()
        valid_fr_ids = [
            fid for fid in valid_fr_ids if 0 <= fid <= temporal_len
        ]  # Ensure valid indices

        # Ensure the number of valid frame indices matches the number of bounding boxes
        if len(valid_fr_ids) != torch.sum(tracklet_indices).item():
            print(
                f"Warning: Mismatch between valid frame indices and bounding boxes for tracklet {n}. Skipping."
            )
            continue

        # Get tracklet_embs
        # print("det_bboxes shape:", det_bboxes.shape)
        # print("tracklet_embs shape:", tracklet_embs.shape)
        # print("tracklet_embs shape:", tracklet_embs)
        tracklet_embs[n, :, valid_fr_ids] = det_bboxes[tracklet_indices, :].permute(
            1, 0
        )
        scores = scores.to(device)
        tracklet_scores[n, 0, valid_fr_ids] = scores[tracklet_indices]

        # Get gt tracklet label
        if stage == "train":
            counter = Counter()
            for idx in torch.nonzero(tracklet_indices).squeeze(1):
                frame_id = det_fr_ids[idx]
                det_bbox = det_bboxes[idx]

                # Find all gt_bboxes in the same frame
                same_frame_gt_indices = torch.nonzero(gt_fr_ids == frame_id).squeeze(1)
                same_frame_gt_bboxes = gt_bboxes[same_frame_gt_indices]

                if same_frame_gt_bboxes.size(0) > 0:
                    # Compute IoU
                    iou = calculate_iou(det_bbox, same_frame_gt_bboxes)

                    # Find the GT with the highest IoU
                    max_iou_idx = torch.argmax(iou)
                    counter[int(gt_ids[same_frame_gt_indices[max_iou_idx]].item())] += 1
                    # tracklet_gt_bboxes[n, :, frame_id] = same_frame_gt_bboxes[
                    #     max_iou_idx, :
                    # ]

            if counter.total() > 0:
                gt_tracklet_labels[n] = counter.most_common(1)[0][0]
            else:
                gt_tracklet_labels[n] = -1

        if stage == "train":
            for det_fr_id in det_fr_ids[tracklet_indices]:
                matching_gt_indices = torch.nonzero(
                    (gt_fr_ids == det_fr_id) & (gt_ids == gt_tracklet_labels[n])
                ).squeeze(1)
                for gt_idx in matching_gt_indices:
                    temporal_position = int(det_fr_id.item())
                    tracklet_gt_bboxes[n, :, temporal_position] = gt_bboxes[gt_idx, :]

    tracklet_mask1 = tracklet_scores
    tracklet_mask2 = tracklet_scores.permute(1, 0, 2)
    tracklet_mask = tracklet_mask1 + tracklet_mask2
    mask_max = torch.max(tracklet_mask, dim=2)[0]
    # print("mask_max", mask_max)
    edge_idx = torch.nonzero(mask_max < 1.5)
    # print("soft_label", soft_label)
    # if soft_label:
    #     edge_idx = torch.nonzero(mask_max < 1.9)
    # else:
    #     edge_idx = torch.nonzero(mask_max < 3.0)
    # edge_idx = torch.nonzero(mask_max < 3.0)
    A = torch.zeros_like(mask_max, device=device)
    A[edge_idx[:, 0], edge_idx[:, 1]] = 1
    # print("edge_idx", edge_idx)
    # print("tracklet_mask", tracklet_mask)

    if stage == "train":
        gt_tracklet_labels = torch.from_numpy(gt_tracklet_labels).float().to(device)
        binary_label = create_binary_label(gt_tracklet_labels)
        tracklet_data = {
            "tracklet_embs": tracklet_embs,
            "tracklet_scores": tracklet_scores,
            "tracklet_labels": gt_tracklet_labels,
            "A": A,
            "binary_label": binary_label,
            "edge_idx": edge_idx,
            "tracklet_gt_embs": tracklet_gt_bboxes,
        }
        # print("tracklet_data", tracklet_data)
    else:
        tracklet_data = {
            "tracklet_embs": tracklet_embs,
            "tracklet_scores": tracklet_scores,
            "A": A,
            "edge_idx": edge_idx,
            "uniq_tracklet_label": uniq_tracklet_label,
        }

    return tracklet_data


def get_tracklet_info_ori(
    tracklet_label,
    obj_ids,
    fr_ids,
    det_embs,
    gt_embs,
    scores,
    temporal_len,
    device,
    stage="train",
):

    uniq_tracklet_label = np.unique(tracklet_label)
    if uniq_tracklet_label[0] == -1:
        uniq_tracklet_label = uniq_tracklet_label[1:]

    N_tracklet = len(uniq_tracklet_label)

    tracklet_embs = torch.zeros(
        N_tracklet, det_embs.shape[1], temporal_len + 1, device=device
    )
    tracklet_scores = torch.zeros(N_tracklet, 1, temporal_len + 1, device=device)
    if stage == "train":
        tracklet_gt_embs = torch.zeros(
            N_tracklet, det_embs.shape[1], temporal_len + 1, device=device
        )
        gt_tracklet_labels = np.zeros((N_tracklet))

    for n in range(N_tracklet):

        # get gt tracklet label
        if stage == "train":
            gt_node_labels = obj_ids[tracklet_label == uniq_tracklet_label[n]]
            tmp_bin_count = np.bincount(
                np.array(gt_node_labels.to("cpu").detach().numpy(), dtype=np.int32)
            )
            gt_tracklet_labels[n] = np.argmax(tmp_bin_count)

        # get tracklet_embs
        tracklet_embs[
            n, :, fr_ids[tracklet_label == uniq_tracklet_label[n]].int().tolist()
        ] = det_embs[tracklet_label == uniq_tracklet_label[n], :].permute(1, 0)
        tracklet_scores[
            n, 0, fr_ids[tracklet_label == uniq_tracklet_label[n]].int().tolist()
        ] = scores[tracklet_label == uniq_tracklet_label[n]]

        if stage == "train":
            tracklet_gt_embs[
                n, :, fr_ids[tracklet_label == uniq_tracklet_label[n]].int().tolist()
            ] = gt_embs[tracklet_label == uniq_tracklet_label[n], :].permute(1, 0)

    tracklet_mask1 = tracklet_scores
    tracklet_mask2 = tracklet_scores.permute(1, 0, 2)
    tracklet_mask = tracklet_mask1 + tracklet_mask2
    mask_max = torch.max(tracklet_mask, dim=2)[0]
    edge_idx = torch.nonzero(mask_max < 1.5)
    A = torch.zeros_like(mask_max, device=device)
    A[edge_idx[:, 0], edge_idx[:, 1]] = 1

    if stage == "train":
        gt_tracklet_labels = torch.from_numpy(gt_tracklet_labels).float().to(device)
        binary_label = create_binary_label(gt_tracklet_labels)
        tracklet_data = {
            "tracklet_embs": tracklet_embs,
            "tracklet_scores": tracklet_scores,
            "tracklet_labels": gt_tracklet_labels,
            "A": A,
            "binary_label": binary_label,
            "edge_idx": edge_idx,
            "tracklet_gt_embs": tracklet_gt_embs,
        }
    else:
        tracklet_data = {
            "tracklet_embs": tracklet_embs,
            "tracklet_scores": tracklet_scores,
            "A": A,
            "edge_idx": edge_idx,
            "uniq_tracklet_label": uniq_tracklet_label,
        }

    return tracklet_data


def associate_tracklet_ori(Dist, non_A, tracklet_labels, thresh=0.3):
    final_labels = tracklet_labels.copy()
    A = non_A.copy()
    A[A > 0] = 1
    Dist[A > 0] = 100
    while True:
        min_dist = np.min(Dist)
        if min_dist > thresh:
            break
        min_idx = np.unravel_index(np.argmin(Dist, axis=None), Dist.shape)
        # print(min_dist)
        # print(min_idx)
        label1 = final_labels[min_idx[0]]
        label2 = final_labels[min_idx[1]]

        final_labels[final_labels == label1] = min(label1, label2)
        final_labels[final_labels == label2] = min(label1, label2)

        Dist[min_idx[0], min_idx[1]] = 100
        Dist[min_idx[1], min_idx[0]] = 100

        A[min_idx[0], min_idx[1]] = 1
        A[min_idx[1], min_idx[0]] = 1

        new_tmp_A = np.sum(A[final_labels == min(label1, label2), :], 0)
        idx = np.where(final_labels == min(label1, label2))[0]
        for k in range(len(idx)):
            A[idx[k], :] = new_tmp_A
            A[:, idx[k]] = new_tmp_A
        Dist[A > 0] = 100

    return final_labels


def associate_tracklet(Dist, non_A, tracklet_labels, thresh=0.3):
    final_labels = tracklet_labels.copy()
    # print("tracklet_labels", tracklet_labels)
    A = non_A.copy()
    A[A > 0] = 1
    Dist[A > 0] = 100

    # 添加字典來記錄每一幀的id
    frame_id_dict = {}

    while True:
        min_dist = np.min(Dist)
        if min_dist > thresh:
            break
        min_idx = np.unravel_index(np.argmin(Dist, axis=None), Dist.shape)

        label1 = final_labels[min_idx[0]]
        label2 = final_labels[min_idx[1]]

        # 檢查兩個軌跡是否有時間重疊
        frames1 = set(np.where(final_labels == label1)[0])
        frames2 = set(np.where(final_labels == label2)[0])

        # 如果有時間重疊，不進行合併
        if frames1.intersection(frames2):
            Dist[min_idx[0], min_idx[1]] = 100
            Dist[min_idx[1], min_idx[0]] = 100
            continue

        # 如果沒有重疊，執行合併
        final_labels[final_labels == label1] = min(label1, label2)
        final_labels[final_labels == label2] = min(label1, label2)

        Dist[min_idx[0], min_idx[1]] = 100
        Dist[min_idx[1], min_idx[0]] = 100

        A[min_idx[0], min_idx[1]] = 1
        A[min_idx[1], min_idx[0]] = 1

        new_tmp_A = np.sum(A[final_labels == min(label1, label2), :], 0)
        idx = np.where(final_labels == min(label1, label2))[0]
        for k in range(len(idx)):
            A[idx[k], :] = new_tmp_A
            A[:, idx[k]] = new_tmp_A
        Dist[A > 0] = 100

    return final_labels


def interp(fr_num, obj_id, bbox, N):
    # 確保輸入數據有效
    if np.any(np.isnan(fr_num)) or np.any(np.isnan(bbox)):
        raise ValueError("Input contains NaN values.")
    if np.any(np.isinf(fr_num)) or np.any(np.isinf(bbox)):
        raise ValueError("Input contains Inf values.")

    fr_num.astype(int)
    obj_id.astype(int)
    new_fr_num = np.empty((0, 1))
    new_obj_id = np.empty((0, 1))
    new_bbox = np.empty((0, 4))

    for i in range(min(obj_id), max(obj_id) + 1):
        cur_fr_num = fr_num[obj_id == i]
        cur_id = obj_id[obj_id == i]
        cur_bbox = bbox[obj_id == i, :]

        if len(cur_fr_num) == 0 or len(cur_fr_num) < N:
            continue

        if len(cur_fr_num) == (max(cur_fr_num) - min(cur_fr_num) + 1):
            new_fr_num = np.append(
                new_fr_num, cur_fr_num.reshape(len(cur_fr_num), 1), axis=0
            )
            new_obj_id = np.append(
                new_obj_id, cur_id.reshape(len(cur_fr_num), 1), axis=0
            )
            new_bbox = np.append(new_bbox, cur_bbox, axis=0)
            continue

        try:
            f_x = interpolate.interp1d(cur_fr_num, cur_bbox[:, 0], kind="linear")
            f_y = interpolate.interp1d(cur_fr_num, cur_bbox[:, 1], kind="linear")
            f_w = interpolate.interp1d(cur_fr_num, cur_bbox[:, 2], kind="linear")
            f_h = interpolate.interp1d(cur_fr_num, cur_bbox[:, 3], kind="linear")
        except ValueError as e:
            print(f"Interpolation error for object {i}: {e}")
            continue

        cur_count = max(cur_fr_num) - min(cur_fr_num) + 1
        interp_fr_num = np.linspace(min(cur_fr_num), max(cur_fr_num), cur_count).astype(
            int
        )
        interp_x = f_x(interp_fr_num).reshape(cur_count, 1)
        interp_y = f_y(interp_fr_num).reshape(cur_count, 1)
        interp_w = f_w(interp_fr_num).reshape(cur_count, 1)
        interp_h = f_h(interp_fr_num).reshape(cur_count, 1)
        interp_bbox = np.concatenate((interp_x, interp_y, interp_w, interp_h), axis=1)

        new_bbox = np.append(new_bbox, interp_bbox, axis=0)
        interp_obj_id = np.full((len(interp_fr_num), 1), i).astype(int)
        new_obj_id = np.append(new_obj_id, interp_obj_id, axis=0)
        new_fr_num = np.append(new_fr_num, interp_fr_num.reshape(cur_count, 1), axis=0)

    return np.squeeze(new_fr_num), np.squeeze(new_obj_id), new_bbox


def test_get_tracklet_info():
    # 模擬輸入數據
    det_ids = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3])
    gt_ids = torch.tensor([10, 10, 10, 20, 20, 30, 30, 30])
    det_fr_ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2])
    gt_fr_ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2])
    det_bboxes = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
            [0.3, 0.3, 0.4, 0.4],
            [0.3, 0.3, 0.4, 0.4],
            [0.2, 0.2, 0.3, 0.3],
        ]
    )
    gt_bboxes = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
            [0.2, 0.2, 0.3, 0.3],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    temporal_len = 3
    device = "cpu"
    stage = "train"

    tracklet_data = get_tracklet_info(
        det_ids=det_ids,
        gt_ids=gt_ids,
        det_fr_ids=det_fr_ids,
        gt_fr_ids=gt_fr_ids,
        det_bboxes=det_bboxes,
        gt_bboxes=gt_bboxes,
        scores=scores,
        temporal_len=temporal_len,
        device=device,
        stage=stage,
    )

    # tracklet_data = get_tracklet_info_ori(
    #     tracklet_label=det_ids,
    #     obj_ids=gt_ids,
    #     fr_ids=det_fr_ids,
    #     det_embs=det_bboxes,
    #     gt_embs=gt_bboxes,
    #     scores=scores,
    #     temporal_len=temporal_len,
    #     device=device,
    #     stage=stage,
    # )

    # 打印結果
    print("Tracklet Embeddings:")
    print(tracklet_data["tracklet_embs"])
    print("Tracklet Scores:")
    print(tracklet_data["tracklet_scores"])
    print("Tracklet Labels:")
    print(tracklet_data["tracklet_labels"])
    print("Adjacency Matrix A:")
    print(tracklet_data["A"])
    print("Binary Label:")
    print(tracklet_data["binary_label"])
    print("Edge Index:")
    print(tracklet_data["edge_idx"])
    print("Tracklet GT Embeddings:")
    print(tracklet_data["tracklet_gt_embs"])


if __name__ == "__main__":
    test_get_tracklet_info()
