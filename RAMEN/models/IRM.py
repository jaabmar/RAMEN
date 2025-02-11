import random

import numpy as np
import torch
from torch.autograd import grad

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class IRM:
    # from https://github.com/claudiashi57/nice/blob/main/src/experiment_synthetic/models.py
    def __init__(self, environments, args):
        self.environments = environments
        self.x_all = torch.cat([x for (x, t, p, y) in environments])
        self.t_all = torch.cat([t for (x, t, p, y) in environments])
        self.y_all = torch.cat([y for (x, t, p, y) in environments])

        self._initialize_model(args)

    def _initialize_model(self, args):
        best_reg, best_err = 0, float('inf')
        x_val, t_val, y_val = self.environments[-1][0], self.environments[-1][1], self.environments[-1][3]

        for reg in [0.1, 0.5]:
            self.train(self.environments[:-1], args, reg=reg)
            y_pred = self._phi(t_val, x_val) @ self.w
            err = ((y_val - y_pred) ** 2).mean().item()

            if err < best_err:
                best_err, best_reg = err, reg
                best_phi0, best_phi1 = self.phi0.clone(), self.phi1.clone()

            print(f'Regularization: {reg}, Error: {err}')

        self.phi0, self.phi1 = best_phi0, best_phi1
        self.y0 = self.x_all @ self.phi0 @ self.w
        self.y1 = self.x_all @ self.phi1 @ self.w

    def _phi(self, t, x):
        return t * (x @ self.phi1) + (1 - t) * (x @ self.phi0)

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)
        self.phi0, self.phi1 = torch.nn.Parameter(torch.eye(dim_x)), torch.nn.Parameter(torch.eye(dim_x))
        self.w = torch.ones(dim_x, 1, requires_grad=True)

        optimizer = torch.optim.Adam([self.phi0, self.phi1], lr=1e-3)
        mse_loss = torch.nn.MSELoss()

        for _ in range(10000):
            penalty, error_sum = 0, 0

            for x_e, t_e, _, y_e in environments:
                error = mse_loss(self._phi(t_e, x_e) @ self.w, y_e)
                penalty += grad(error, self.w, create_graph=True)[0].pow(2).mean()
                error_sum += error

            optimizer.zero_grad()
            (reg * error_sum + (1 - reg) * penalty).backward()
            optimizer.step()

    def ite(self):
        return (self.y1 - self.y0).reshape(-1, 1)

    def accuracy(self):
        y_pred = self.y1 * self.t_all + self.y0 * (1 - self.t_all)
        return ((self.y_all - y_pred) ** 2).mean()

    def att(self):
        return self.ite()[self.t_all == 1].mean()

    def ate(self, n_env):
        ates = []
        for e in range(n_env):
            X, T = self.environments[e][0], self.environments[e][1]
            y0, y1 = X @ self.phi0 @ self.w, X @ self.phi1 @ self.w
            ates.append((y1 - y0).mean())
        return ates
