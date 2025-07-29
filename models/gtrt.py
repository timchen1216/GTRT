import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as einops

from models.transformer import EncoderBlock, DecoderBlock, PositionalEmbedding
from models.ffn import FFN


class GTRT(nn.Module):
    def __init__(
        self,
        temporal_length: int,
        num_id_vocabulary: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: torch.device,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        max_length: int = 10000,
    ):
        super().__init__()
        self.device = device
        # self.history_tracklet_emb = torch.zeros(
        #     (num_id_vocabulary, temporal_length, emb_dim), device=self.device
        # )
        self.num_id_vocabulary = num_id_vocabulary
        self.num_layers = num_layers
        self.num_bbox_layers = 2
        self.bbox_dim = 4
        self.bbox_emb_dim = 32
        self.emb_dim = temporal_length * self.bbox_emb_dim
        self.ffn_dim_ratio = 2  # Ratio for FFN dimension expansion
        # self.embedding = nn.Embedding(track_size, dim, padding_idx=4)
        self.positional_encoding1 = PositionalEmbedding(self.bbox_emb_dim, max_length)
        self.positional_encoding2 = PositionalEmbedding(
            self.bbox_emb_dim * temporal_length, max_length
        )
        self.last_pred_labels = None

        self.tracklet_features = None
        self.tracklet_id_labels = []
        # self.tracklet_dic = nn.Parameter(torch.zeros((num_id_vocabulary, emb_dim)))
        self.word_to_embed = nn.Linear(
            self.num_id_vocabulary + 1,
            self.emb_dim,
            bias=False,
            device=self.device,
        )

        self.bbox_enc_layers = nn.ModuleList()
        for _ in range(self.num_bbox_layers):
            self.bbox_enc_layers.append(
                EncoderBlock(self.bbox_emb_dim, self.bbox_emb_dim * 2, num_heads=1)
            )

        self.track_enc_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.track_enc_layers.append(
                EncoderBlock(self.emb_dim, hidden_dim, num_heads, dropout_prob)
            )
        self.dec_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.dec_layers.append(
                DecoderBlock(
                    2 * self.emb_dim,
                    hidden_dim,
                    2 * self.emb_dim,
                    num_heads,
                    dropout_prob,
                )
            )
        # Add new linear layer for feature dimension expansion
        self.box_embedding = nn.Linear(self.bbox_dim, self.bbox_emb_dim)
        self.adapter = FFN(
            d_model=self.emb_dim,
            d_ffn=emb_dim * self.ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(self.emb_dim)
        self.embed_to_word = nn.Linear(
            self.emb_dim, self.num_id_vocabulary + 1, bias=False
        )
        # self.embed_to_word_layers = nn.ModuleList(
        #     [self.embed_to_word for _ in range(self.num_layers)]
        # )

        self.initialize_weights()
        self.set_device()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if "embed_to_word" in name or "word_to_embed" in name:
                    # 對輸出層使用更小的初始化
                    nn.init.xavier_uniform_(param, gain=0.01)
                elif "scale" in name:
                    # RMSNorm 的 scale 參數
                    nn.init.constant_(param, 1.0)
                else:
                    nn.init.xavier_uniform_(param)
            else:
                if "bias" in name:
                    nn.init.constant_(param, 0.0)

    def set_device(self):
        for m in self.modules():
            m = m.to(self.device)

    def forward(self, data, teacher_forcing_prob=1.0, stage="train"):
        # for k in data.keys():
        #     print(
        #         f"{k}: {data[k].shape if hasattr(data[k], 'shape') else type(data[k])}"
        #     )

        # load data
        tracklet_bbox = data["tracklet_bbox"].clone()  # [B, N, D, T]
        tracklet_mask = data["tracklet_mask"].clone()  # [B, N, D, T]
        tracklet_id_mask = data["tracklet_id_mask"].clone()  # [B, N]

        history_bbox = data["history_bbox"].clone()  # [B, N, D, T]
        history_mask = data["history_mask"].clone()  # [B, N, D, T]
        history_labels = data["history_labels"].clone()  # [B, N]
        history_id_mask = data["history_id_mask"].clone()  # [B, N]
        B, N, D, T = tracklet_bbox.shape
        if stage == "train" or stage == "val":
            tracklet_labels = data["tracklet_labels"].clone()  # [B, N]
            tracklet_gt_bbox = data["gt_bboxes"].clone()  # [B, N, D, T]
            tracklet_gt_mask = data["gt_mask"].clone()  # [B, N, D, T]
            gt_binary_label = data["binary_label"].clone()  # [B,N,N]
            time_window = data["time_window"].clone()  # [B, 2]
            history_time_window = data["history_time_window"].clone()  # [B, 2]
            # print("history_time_window", history_time_window)

            # Teacher Forcing & Shuffle
            prob = random.random() if stage == "train" else 1
            external_last_pred = data.get("external_last_pred", None)
            external_last_masks = data.get("external_last_masks", None)
            # print("external_last_pred", external_last_pred)
            # print("external_last_masks", external_last_masks)
            # print("teacher_forcing_prob", teacher_forcing_prob)
            if prob >= teacher_forcing_prob and external_last_pred is not None:
                # print("use predict")
                _B, _N = external_last_pred.shape

                history_labels = -torch.ones(
                    (B, N), dtype=history_labels.dtype, device=history_labels.device
                )
                history_id_mask = torch.zeros(
                    (B, N), dtype=history_id_mask.dtype, device=history_id_mask.device
                )

                copy_size = min(N, _N)

                history_labels[:, :copy_size] = external_last_pred[:, :copy_size]
                history_id_mask[:, :copy_size] = external_last_masks[:, :copy_size]

            for b in range(B):
                if history_time_window[b][0] == 1.0:
                    # print("seq start")
                    history_labels[b] = data["history_labels"][b]
                    history_id_mask[b] = data["history_id_mask"][b]

            # if stage == "train":
            #     for b in range(B):
            #         shuffle_indices = torch.randperm(N)
            #         inverse_shuffle_indices = torch.argsort(shuffle_indices)
            #         tracklet_bbox[b] = tracklet_bbox[b][shuffle_indices]
            #         tracklet_mask[b] = tracklet_mask[b][shuffle_indices]
            #         tracklet_labels[b] = tracklet_labels[b][shuffle_indices]
            #         tracklet_id_mask[b] = tracklet_id_mask[b][shuffle_indices]

            #         # history_shuffle_indices = torch.randperm(N)
            #         history_bbox[b] = history_bbox[b][shuffle_indices]
            #         history_mask[b] = history_mask[b][shuffle_indices]
            #         history_labels[b] = history_labels[b][shuffle_indices]
            #         history_id_mask[b] = history_id_mask[b][shuffle_indices]

        tracklet_bbox = einops.rearrange(tracklet_bbox, "b n d t -> n b t d")
        history_bbox = einops.rearrange(history_bbox, "b n d t -> n b t d")
        tracklet_emb = self.box_embedding(tracklet_bbox)
        history_emb = self.box_embedding(history_bbox)

        unknown_features = torch.zeros((N, B, T, self.bbox_emb_dim), device=self.device)
        history_features = torch.zeros((N, B, T, self.bbox_emb_dim), device=self.device)

        for i, (x, y) in enumerate(zip(tracklet_emb, history_emb)):
            # Expand feature dimension first
            x = self.positional_encoding1(x)
            y = self.positional_encoding1(y)
            unknown_features[i] = x
            history_features[i] = y

        tracklet_mask = einops.rearrange(tracklet_mask, "b n d t -> n b t d")
        history_mask = einops.rearrange(history_mask, "b n d t -> n b t d")
        tracklet_mask = tracklet_mask.any(dim=-1)  # [N, B, T]
        history_mask = history_mask.any(dim=-1)  # [N, B, T]

        unknown_list = []
        history_list = []
        for n in range(N):
            unk_feat = unknown_features[n]
            hist_feat = history_features[n]
            for layer in self.bbox_enc_layers:
                unk_feat = layer(unk_feat, tracklet_mask[n])
                hist_feat = layer(hist_feat, history_mask[n])
            unknown_list.append(unk_feat)
            history_list.append(hist_feat)

        unknown_features = torch.stack(unknown_list, dim=0)
        history_features = torch.stack(history_list, dim=0)

        unknown_features = einops.rearrange(unknown_features, "n b t d -> b n (t d)")
        history_features = einops.rearrange(history_features, "n b t d -> b n (t d)")

        unknown_features = self.positional_encoding2(unknown_features)
        history_features = self.positional_encoding2(history_features)

        for layer in self.track_enc_layers:
            unknown_features = layer(unknown_features, mask=tracklet_id_mask)
            history_features = layer(history_features, mask=history_id_mask)

        unknown_id_embeds = self.generate_empty_id_embed(
            unknown_features=unknown_features
        )  # [B, N, T*D]

        tracklet_id_embeds = self.id_label_to_embed(
            id_labels=history_labels
        )  # [B, N, emb_dim]
        # print("unknown_id_embeds shape:", unknown_id_embeds.shape)
        # print("tracklet_id_embeds shape:", tracklet_id_embeds.shape)
        unknown_embedding = torch.cat(
            [unknown_features, unknown_id_embeds], dim=-1
        )  # [B, N, 2*emb_dim]
        tracklet_embedding = torch.cat(
            [history_features, tracklet_id_embeds], dim=-1
        )  # [B, N, 2*emb_dim]

        # print("tracklet_id_mask", tracklet_id_mask.shape)
        # print("history_id_mask", history_id_mask.shape)

        for layer in self.dec_layers:
            unknown_embedding = layer(
                x=unknown_embedding,
                memory=tracklet_embedding,
                mask=tracklet_id_mask,
                memory_mask=history_id_mask,
            )

        # print("unknown_embedding shape:", unknown_embedding.shape)  # [B, N, 2*emb_dim]
        # _unknown_id_prob = self.embed_to_word(unknown_embedding[..., -self.emb_dim :])
        features = unknown_embedding[..., -self.emb_dim :]

        features = self.output_norm(features)
        logits = self.embed_to_word(features)

        logits = torch.clamp(logits, min=-10, max=10)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        _unknown_id_prob = probs / probs.sum(dim=-1, keepdim=True)
        _unknown_id_labels = torch.argmax(_unknown_id_prob, dim=-1)
        # print("_unknown_id_labels", _unknown_id_labels.shape)
        # del (
        #     unknown_features,
        #     history_features,
        #     unknown_id_embeds,
        #     tracklet_id_embeds,
        #     unknown_embedding,
        #     tracklet_embedding,
        #     tracklet_one_hot_labels,
        # )
        if stage == "train" or stage == "val":

            tracklet_one_hot_labels = self.label_to_one_hot(
                tracklet_labels, self.num_id_vocabulary + 1
            )
            bce_loss = self.extract_and_compute_loss(
                _unknown_id_prob, tracklet_one_hot_labels, tracklet_id_mask.clone()
            )
            # if stage == "train":
            #     for b in range(B):
            #         _unknown_id_labels[b] = _unknown_id_labels[b][
            #             inverse_shuffle_indices
            #         ]
            #         tracklet_id_mask[b] = tracklet_id_mask[b][inverse_shuffle_indices]
            # print("bce_loss shape:", bce_loss)  # [B, N, K]

            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            return {
                "bce_loss": bce_loss,
                "pred_labels": _unknown_id_labels,
                "pred_masks": tracklet_id_mask,
            }
        elif stage == "test":

            return {
                "unknown_id_prob": _unknown_id_prob,
            }

    def label_to_one_hot(
        self, labels: torch.Tensor, n_classes: int, dtype=torch.float32
    ):
        # 創建 one-hot tensor，初始化為全0
        one_hot = torch.zeros(
            labels.shape + (n_classes,), dtype=dtype, device=labels.device
        )

        # 找到有效的標籤（>= 0 且 < n_classes）
        valid_mask = (labels >= 0) & (labels < n_classes)

        if valid_mask.any():
            # 只對有效標籤進行 one-hot 編碼
            valid_labels = labels[valid_mask]
            eye_matrix = torch.eye(n_classes, dtype=dtype, device=labels.device)
            one_hot[valid_mask] = eye_matrix[valid_labels]

        # 無效標籤（如 -1）會保持為全0向量
        return one_hot

    def id_label_to_embed(self, id_labels):
        id_words = self.label_to_one_hot(id_labels, self.num_id_vocabulary + 1)
        id_embeds = self.word_to_embed(id_words.to(self.device))
        return id_embeds

    def generate_empty_id_embed(self, unknown_features):
        _shape = unknown_features.shape[:-1]
        empty_id_labels = self.num_id_vocabulary * torch.ones(
            _shape, dtype=torch.int64, device=unknown_features.device
        )
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds

    def extract_and_compute_loss(self, predictions, targets, mask):
        """
        最簡潔的實現：提取有效數據並計算 loss
        """
        # 找到所有有效位置
        valid_mask = mask.bool()

        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 直接使用 mask 索引提取所有有效數據
        valid_preds = predictions[valid_mask]  # [total_valid, K]
        valid_targets = targets[valid_mask]  # [total_valid, K]

        # print("valid_preds", valid_preds.shape)
        # print("valid_targets", valid_targets.shape)

        return F.binary_cross_entropy(valid_preds, valid_targets, reduction="mean")
