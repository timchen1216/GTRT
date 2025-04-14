import torch
import torch.nn.functional as F

import numpy as np
import head_utils


class DetGraphEmb(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(DetGraphEmb, self).__init__()

        self.device = device

        box_input_channel = 4
        num_layers = 8
        fea_dim = 128

        box_arch = []
        att_arch = []
        for l in range(num_layers):
            box_arch.append([fea_dim, fea_dim])
            att_arch.append([fea_dim, 1])

        self.box_layers = torch.nn.ModuleList([])
        self.att_layers = torch.nn.ModuleList([])

        for l in range(len(box_arch)):
            tmp_box_layers = torch.nn.ModuleList([])
            tmp_att_layers = torch.nn.ModuleList([])
            if l == 0:
                tmp_box_layers.append(
                    torch.nn.Linear(box_input_channel, box_arch[l][0])
                )
                tmp_box_layers.append(torch.nn.Linear(box_arch[l][0], box_arch[l][1]))
                tmp_att_layers.append(
                    torch.nn.Linear(box_input_channel, att_arch[l][0])
                )
                tmp_att_layers.append(torch.nn.Linear(att_arch[l][0], att_arch[l][1]))
            else:
                tmp_box_layers.append(
                    torch.nn.Linear(box_arch[l - 1][1], box_arch[l][0])
                )
                tmp_box_layers.append(torch.nn.Linear(box_arch[l][0], box_arch[l][1]))
                tmp_att_layers.append(
                    torch.nn.Linear(box_arch[l - 1][1], att_arch[l][0])
                )
                tmp_att_layers.append(torch.nn.Linear(att_arch[l][0], att_arch[l][1]))
            self.box_layers.append(tmp_box_layers)
            self.att_layers.append(tmp_att_layers)

    def forward(self, x_box, A, edge_idx, binary_label=None):

        N = len(x_box)
        int_att_loss = []
        for l in range(len(self.box_layers)):
            prev_X_box = x_box

            # attention
            X1 = x_box[edge_idx[0, :].tolist()]
            X2 = x_box[edge_idx[1, :].tolist()]
            tmp_att = torch.abs(X1 - X2)

            tmp_att = F.leaky_relu(self.att_layers[l][0](tmp_att))
            tmp_att = F.sigmoid(self.att_layers[l][1](tmp_att))
            att = torch.zeros(N, N, device=self.device)
            att[edge_idx[0, :].tolist(), edge_idx[1, :].tolist()] = tmp_att[:, 0]
            att = att + torch.eye(N, device=self.device)

            if binary_label is not None:
                if (
                    torch.min(A * att).item() < 0.0
                    or torch.max(A * att).item() > 1.0
                    or torch.min(A * binary_label).item() < 0.0
                    or torch.max(A * binary_label).item() > 1.0
                ):
                    import pdb

                    pdb.set_trace()
                if (
                    torch.sum(torch.isnan((A * att))).item() > 0
                    or torch.sum(torch.isnan((A * binary_label))).item() > 0
                ):
                    import pdb

                    pdb.set_trace()
                int_att_loss.append(F.binary_cross_entropy(A * att, A * binary_label))

            update_A = A + torch.eye(N, device=self.device)
            update_A = att * update_A
            D = torch.sum(update_A, 1, keepdim=True)
            update_A = 1.0 / D * update_A

            x_box = self.box_layers[l][0](x_box)
            x_box = F.leaky_relu(x_box)
            x_box = self.box_layers[l][1](x_box)
            # import pdb; pdb.set_trace()
            x_box = torch.mm(update_A, x_box)
            x_box = F.leaky_relu(x_box)
            x_box = F.normalize(x_box, p=2, dim=1)

            if x_box.shape == prev_X_box.shape and l < len(self.box_layers) - 1:
                x_box = x_box + prev_X_box
            # print("x_box.shape", x_box.shape)
        return x_box, int_att_loss


def build_adj_graph(fr_ids, bbox):
    # Frame connectivity
    N = len(fr_ids)
    F1 = torch.unsqueeze(fr_ids, 1)  # Shape: (N, 1)
    F2 = torch.unsqueeze(fr_ids, 0)  # Shape: (1, N)
    A = torch.abs(F1 - F2)  # Temporal adjacency (e.g., frame difference of 1)
    A[A != 1] = 0  # Only connect nodes in consecutive frames

    # Extract edge indices
    edge_idx = torch.nonzero(A, as_tuple=False).t()  # Shape: (2, E)
    return edge_idx, A


def get_adj_label(A, obj_ids):
    N = len(obj_ids)
    F1 = torch.unsqueeze(obj_ids, 1)
    F2 = torch.unsqueeze(obj_ids, 0)
    D = torch.abs(F1 - F2)
    is_pos = (D == 0) * A
    is_neg = (D != 0) * A
    return is_pos, is_neg


def test():
    fr_ids = torch.tensor([0, 0, 1, 1, 2])
    bbox = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12], [3, 3, 13, 13], [4, 4, 14, 14]]
    )
    edge_idx, A = build_adj_graph(fr_ids, bbox)
    tmp_idx = torch.nonzero(edge_idx[0, :] < edge_idx[1, :])
    transf_edge_idx = edge_idx[:, tmp_idx[:, 0]]
    dist = torch.norm(
        bbox[transf_edge_idx[0, :], :] - bbox[transf_edge_idx[1, :], :],
        dim=1,
    )

    print("Edge Index:")
    print(edge_idx)
    print("Adjacency Matrix:")
    print(A)
    print("tmp_idx", tmp_idx.shape)
    print("tmp_idx", tmp_idx)
    print("transf_edge_idx", transf_edge_idx.shape)
    print("transf_edge_idx", transf_edge_idx)
    print("Distance:")
    print(dist)


if __name__ == "__main__":
    test()
