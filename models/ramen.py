import concurrent.futures
import time
from functools import partial
from itertools import chain, combinations

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from models.models_utils import (XGB_PARAMS, compute_cross_statistic,
                                 construct_cross_kernel_matrix)


def generate_subsets(data):
    return chain.from_iterable(combinations(data, r) for r in range(1, len(data) + 1))


class Ramen:
    def __init__(self, input_dim: int, n_env: int, use_xgboost=False, logger=None):
        self.input_dim = input_dim
        self.n_env = n_env
        self.use_xgboost = use_xgboost
        self.logger = logger

    def compute_stat(self, x, psi):
        K = construct_cross_kernel_matrix(x)
        return compute_cross_statistic(psi, K)

    def compute_subset(self, X, Y, T):
        start_time = time.time()
        X, T, Y = map(torch.from_numpy, (X, T, Y))

        subsets = list(generate_subsets(range(self.input_dim)))
        partial_loss = partial(self.compute_loss, X=X, Y=Y, T=T)

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(partial_loss, subsets))

        # Sort results by loss value
        sorted_results = sorted(results, key=lambda x: x[0])
        best_subset = sorted_results[0][1]

        if self.logger:
            for loss, subset, loss_t, loss_y in sorted_results:
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Time: {elapsed_time:.5f}s, Subset: {subset}, "
                    f"T-invariance Loss: {loss_t:.5f}, Y-invariance Loss: {loss_y:.5f}, Min Loss: {loss:.5f}"
                )
            self.logger.info(f"Total time for compute_subset: {time.time() - start_time:.5f} seconds")

        return best_subset

    def compute_loss(self, subset, X, Y, T):
        X_pool = X.reshape(-1, X.shape[2])[:, subset]
        T_pool, Y_pool = T.reshape(-1), Y.reshape(-1)

        model_t, model_y1, model_y0 = self.initialize_models()

        model_t.fit(X_pool, T_pool)
        model_y1.fit(X_pool[T_pool == 1], Y_pool[T_pool == 1])
        model_y0.fit(X_pool[T_pool == 0], Y_pool[T_pool == 0])

        loss_t, loss_y = self.calculate_losses(X, Y, T, subset, model_t, model_y1, model_y0)

        return min(loss_t, loss_y), subset, loss_t, loss_y

    def initialize_models(self):
        if self.use_xgboost:
            return (
                XGBClassifier(**XGB_PARAMS),
                XGBRegressor(**XGB_PARAMS),
                XGBRegressor(**XGB_PARAMS)
            )
        else:
            return (
                LogisticRegression(max_iter=500),
                LinearRegression(),
                LinearRegression()
            )

    def calculate_losses(self, X, Y, T, subset, model_t, model_y1, model_y0):
        loss_t, loss_y = [], []

        for e in range(X.shape[0]):
            X_e, T_e, Y_e = X[e, :, subset], T[e, :], Y[e, :]

            psi_t = T_e - model_t.predict_proba(X_e)[:, 1]
            loss_t.append(self.compute_stat(X_e, psi_t))

            psi_y1 = Y_e[T_e == 1] - model_y1.predict(X_e[T_e == 1])
            psi_y0 = Y_e[T_e == 0] - model_y0.predict(X_e[T_e == 0])

            loss_y.extend([
                self.compute_stat(X_e[T_e == 1], psi_y1),
                self.compute_stat(X_e[T_e == 0], psi_y0)
            ])

        return np.max(loss_t), np.max(loss_y)
