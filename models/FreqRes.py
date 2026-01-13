"""
https://github.com/ZOF-pt/SARC/tree/main

@inproceedings{sarc,
  title={Spectral-Aware Reservoir Computing for Fast and Accurate Time Series Classification},
  author={Liu, Shikang and Wei, Chuyang and Zhou, Xiren and Chen, Huanhuan},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
"""

import torch
import numpy as np
import torch.nn as nn


def get_res(hidden_dim, radius, connectivity):
    w = np.random.randn(hidden_dim, hidden_dim)
    mask = np.random.choice([0, 1], size=(hidden_dim, hidden_dim), p=[1 - connectivity, connectivity])
    w = w * mask
    max_lambda = max(abs(np.linalg.eig(w)[0]))
    return radius * w / max_lambda


class FreqBiESN(nn.Module):
    def __init__(self, input_dim, periods, hidden_dim=10, spectral_radius=(0.8, 0.8), regular=1., leaky=0.):
        super(FreqBiESN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.periods = periods
        self.radius = spectral_radius
        self.connectivity = min(1., 10 / hidden_dim)
        self.leaky = leaky

        self.W_in = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim), requires_grad=False)
        W_fs = torch.from_numpy(get_res(hidden_dim, self.radius[0], self.connectivity).astype(np.float32))
        W_fc = [torch.from_numpy(get_res(hidden_dim, self.radius[1], self.connectivity).astype(np.float32)) for _ in periods]
        W_bs = torch.from_numpy(get_res(hidden_dim, self.radius[0], self.connectivity).astype(np.float32))
        W_bc = [torch.from_numpy(get_res(hidden_dim, self.radius[1], self.connectivity).astype(np.float32)) for _ in periods]
        self.W_fs = nn.Parameter(W_fs, requires_grad=False)
        self.W_fc = nn.Parameter(torch.stack(W_fc).unsqueeze(0), requires_grad=False)
        self.W_bs = nn.Parameter(W_bs, requires_grad=False)
        self.W_bc = nn.Parameter(torch.stack(W_bc).unsqueeze(0), requires_grad=False)
        self.regular = nn.Parameter(regular * torch.eye(4 * self.hidden_dim).unsqueeze(0), requires_grad=False)

    def compute_state(self, x_transformed, W_s, W_c):
        batch_size, length, _ = x_transformed.shape
        x_transformed = x_transformed.unsqueeze(1)
        k = len(self.periods)
        periods = torch.tensor(self.periods)
        W_c_exp = W_c.expand(batch_size, k, self.hidden_dim, self.hidden_dim)
        h_history = torch.zeros(batch_size, k, length, self.hidden_dim, device=x_transformed.device)
        for t in range(length):
            row_indices = torch.arange(k)
            col_indices = t - periods
            h_pre_cyc = h_history[:, row_indices, col_indices, :].unsqueeze(dim=2)
            h_t = x_transformed[:, :, t, :] + torch.matmul(h_history[:, :, t-1, :], W_s) + torch.matmul(h_pre_cyc, W_c_exp).squeeze(2)
            h_history[:, :, t, :] = (1 - self.leaky) * torch.tanh(h_t) + self.leaky * h_history[:, :, t-1, :]
        return h_history

    def get_feature(self, h_history, x):
        feature_vectors = []
        batch_size, length, _ = x.size()
        for idx, period in enumerate(self.periods):
            H = torch.cat([h_history[:, idx, period - 1:length - 1, :], h_history[:, idx, :length - period, :]], dim=2)
            X = x[:, period:, :]
            regular = self.regular
            Ht = H.transpose(1, 2)
            HtH = torch.bmm(Ht, H)
            HtX = torch.bmm(Ht, X)
            W_out = torch.linalg.solve(HtH + regular, HtX)
            feature_vectors.append(W_out.flatten(start_dim=1))
        features = torch.cat(feature_vectors, dim=1)
        features = torch.cat([features, torch.max(h_history, dim=2)[0].flatten(start_dim=1)], dim=1)
        return features

    def forward(self, x, count_nan=None):
        batch_size, length, _ = x.size()
        x_transformed = torch.matmul(x, self.W_in)
        h_f = self.compute_state(x_transformed, self.W_fs, self.W_fc)
        h_b = self.compute_state(x_transformed.flip(dims=[1]), self.W_bs, self.W_bc).flip(dims=[2])
        h = torch.cat([h_f, h_b], dim=3)

        if count_nan is not None:
            range_tensor = torch.arange(length, device=h.device).view(1, 1, length, 1)
            count_nan_exp = count_nan.view(batch_size, 1, 1, 1)
            mask = range_tensor < count_nan_exp
            mask = mask.expand(h.shape)
            h = h.masked_fill(mask, 0)

        features = self.get_feature(h, x)
        return features.cpu().numpy()



