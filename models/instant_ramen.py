import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.models_utils import (compute_cross_statistic,
                                 construct_cross_kernel_matrix)
from models.ramen import Ramen


class GumbelGate(nn.Module):
    def __init__(self, input_dim, init_offset=0, device=None):
        super(GumbelGate, self).__init__()
        self.logits = nn.Parameter(
            torch.rand(input_dim, device=device) * 1e-5 + init_offset
        )
        self.input_dim = input_dim
        self.device = device

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def generate_mask(self, temperature):
        gumbel_sample = (
            self.logits
            + self.sample_gumbel(self.logits.shape)
            - self.sample_gumbel(self.logits.shape)
        )
        mask = torch.sigmoid(gumbel_sample / temperature)
        return torch.reshape(mask, (1, self.input_dim))


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_activation=None):
        super(NeuralNet, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.output_activation = output_activation

    def forward(self, x):
        logits = self.fc(x)
        if self.output_activation:
            logits = self.output_activation(logits)
        return logits


class InstantRamen:
    def __init__(
        self,
        input_dim,
        n_env,
        init_temp=1.0,
        final_temp=0.001,
        anneal_iter=100,
        learning_rate=1e-3,
        anneal_rate=0.95,
        n_epochs=500,
        patience=50,
        use_xgboost=False,
        logger=None,
    ):
        super(InstantRamen, self).__init__()
        self.input_dim = input_dim
        self.n_env = n_env
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.anneal_iter = anneal_iter
        self.anneal_rate = anneal_rate
        self.n_epochs = n_epochs
        self.patience = patience
        self.use_xgboost = use_xgboost
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mu_t = NeuralNet(input_dim).to(self.device)
        self.mu_y1 = NeuralNet(input_dim, output_activation=nn.Sigmoid()).to(
            self.device
        )
        self.mu_y0 = NeuralNet(input_dim, output_activation=nn.Sigmoid()).to(
            self.device
        )
        self.set_t = GumbelGate(input_dim, device=self.device)
        self.set_y = GumbelGate(input_dim, device=self.device)
        self.lr = learning_rate

    def compute_subset(self, X, Y, T):
        X, Y, T = map(lambda d: torch.from_numpy(d).float().to(self.device), (X, Y, T))

        mask_t = self.train_t(X, T)
        mask_y = self.train_y(X, Y, T)

        subset_t = torch.where(mask_t == 1)[1]
        subset_y = torch.where(mask_y == 1)[1]

        ramen = Ramen(
            self.input_dim, self.n_env, use_xgboost=self.use_xgboost, logger=self.logger
        )
        loss_t = (
            ramen.compute_loss(subset_t.cpu(), X.cpu(), Y.cpu(), T.cpu())[0]
            if len(subset_t) > 0
            else np.inf
        )
        loss_y = (
            ramen.compute_loss(subset_y.cpu(), X.cpu(), Y.cpu(), T.cpu())[0]
            if len(subset_y) > 0
            else np.inf
        )

        if self.logger:
            self.logger.info(f"Subset T: {subset_t.tolist()}, Loss T: {loss_t:.5f}")
            self.logger.info(f"Subset Y: {subset_y.tolist()}, Loss Y: {loss_y:.5f}")

        selected_subset = subset_t if loss_t < loss_y else subset_y
        selected_loss = min(loss_t, loss_y)

        if self.logger:
            self.logger.info(f"Selected Subset: {selected_subset.tolist()}, Loss: {selected_loss:.5f}")

        return selected_subset.cpu().numpy(), selected_loss

    def compute_loss_y(self, X, Y, T, mask):
        losses = 0
        for e in range(self.n_env):
            X_e = X[e] * mask
            T_e = T[e, :]
            Y_e = Y[e, :]

            K = construct_cross_kernel_matrix(X_e[T_e == 1])
            psi = Y_e[T_e == 1] - self.mu_y1(X_e[T_e == 1]).reshape(-1)
            losses += compute_cross_statistic(psi, K)

            K = construct_cross_kernel_matrix(X_e[T_e == 0])
            psi = Y_e[T_e == 0] - self.mu_y0(X_e[T_e == 0]).reshape(-1)
            losses += compute_cross_statistic(psi, K)

        return losses.mean()

    def compute_loss_t(self, X, T, mask):
        losses = 0
        for e in range(self.n_env):
            X_e = X[e] * mask
            T_e = T[e, :]
            K = construct_cross_kernel_matrix(X_e)
            psi = T_e - self.mu_t(X_e).reshape(-1)
            losses += compute_cross_statistic(psi, K)

        return losses.mean()

    def train_t(self, X, T):
        tau = self.init_temp
        optimizer_set_t = optim.Adam(self.set_t.parameters(), lr=self.lr)
        optimizer_mu_t = optim.SGD(self.mu_t.parameters(), lr=0.01)

        best_loss_t = float("inf")
        epochs_no_improve = 0

        for epoch in tqdm(range(self.n_epochs)):
            if (epoch + 1) % self.anneal_iter == 0:
                tau = max(self.final_temp, tau * self.anneal_rate)

            optimizer_set_t.zero_grad()
            optimizer_mu_t.zero_grad()

            mask_t = self.set_t.generate_mask(tau)
            loss_t = self.compute_loss_t(X, T, mask_t)

            loss_t.backward()
            optimizer_set_t.step()
            optimizer_mu_t.step()

            if loss_t < best_loss_t:
                best_loss_t, epochs_no_improve = loss_t, 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        return self.set_t.generate_mask(self.final_temp)

    def train_y(self, X, Y, T):
        tau = self.init_temp
        optimizer_set_y = optim.Adam(self.set_y.parameters(), lr=self.lr)
        optimizer_mu_y1 = optim.SGD(self.mu_y1.parameters(), lr=0.01)
        optimizer_mu_y0 = optim.SGD(self.mu_y0.parameters(), lr=0.01)

        best_loss_y = float("inf")
        epochs_no_improve = 0

        for epoch in tqdm(range(self.n_epochs)):
            if (epoch + 1) % self.anneal_iter == 0:
                tau = max(self.final_temp, tau * self.anneal_rate)

            optimizer_set_y.zero_grad()
            optimizer_mu_y1.zero_grad()
            optimizer_mu_y0.zero_grad()

            mask_y = self.set_y.generate_mask(tau)
            loss_y = self.compute_loss_y(X, Y, T, mask_y)

            loss_y.backward()
            optimizer_set_y.step()
            optimizer_mu_y1.step()
            optimizer_mu_y0.step()

            if loss_y < best_loss_y:
                best_loss_y, epochs_no_improve = loss_y, 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        return self.set_y.generate_mask(self.final_temp)
