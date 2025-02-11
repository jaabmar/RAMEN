import torch

XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 6,
}


def gaussian_kernel(xi, xj, sigma):
    """Gaussian kernel function."""
    dist_sq = torch.cdist(xi, xj, p=2) ** 2
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def compute_linear_stat(x, psi):
    num_pairs = min(len(x[::2]), len(x[1::2]))
    x1, x2 = x[::2][:num_pairs], x[1::2][:num_pairs]

    dist = torch.norm(x1 - x2, dim=1)
    sigma = dist.median()

    kernel = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    h = psi[::2][:num_pairs] * psi[1::2][:num_pairs] * kernel

    stat = h.mean().abs() / (h.std() + 1e-8)
    return stat


def construct_kernel_matrix(x):
    n = x.shape[0]
    kernel_param = torch.cdist(x, x).flatten().median()

    x_i, x_j = x.unsqueeze(1), x.unsqueeze(0)
    kernel_matrix = gaussian_kernel(x_i, x_j, kernel_param)

    return kernel_matrix.reshape(n, n)


def construct_cross_kernel_matrix(x):
    m = x.shape[0] // 2

    x_i, x_j = x[:m], x[m:]
    kernel_param = torch.cdist(x, x).flatten().median()

    return gaussian_kernel(x_i, x_j, kernel_param)


def compute_cross_statistic(psi, K):
    m1, m2 = K.shape
    idx_i, idx_j = torch.meshgrid(torch.arange(m1), torch.arange(m1, m1 + m2), indexing='ij')

    psi_i, psi_j = psi[idx_i.flatten()], psi[idx_j.flatten()]
    h = psi_i * psi_j * K.flatten()

    fhs = h.view(m1, m2).mean(dim=1)
    U = fhs.mean() / (fhs.std() + 1e-8)

    return torch.abs(U)


def compute_u_statistic(psi, K):
    n = K.shape[0]
    psi_outer = psi.view(-1, 1) @ psi.view(1, -1)

    h = psi_outer * K * (1 - torch.eye(n))
    loss = h.mean()

    return torch.abs(loss)
