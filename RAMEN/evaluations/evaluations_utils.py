
import itertools

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from RAMEN.models.instant_ramen import InstantRamen
from RAMEN.models.IRM import IRM

# Constants
EPSILON = 0.025

XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 6,
}

PARAM_GRID = {
    'learning_rate': [0.01, 0.1, 0.001],
    'init_temp': [0.5, 0.8, 1.0],
    'anneal_rate': [0.9, 0.95, 0.99]
}

PARAM_COMBINATIONS = list(itertools.product(
    PARAM_GRID['learning_rate'],
    PARAM_GRID['init_temp'],
    PARAM_GRID['anneal_rate']
))


def log_best_params(result, logger):
    logger.info("Best parameters: Learning Rate: %s, Initial Temperature: %s, Anneal Rate: %s, Lowest Loss: %s",
                result['learning_rate'], result['init_temp'], result['anneal_rate'], result['loss'])


def pool_data(data, dim):
    return np.asarray([env[dim] for env in data])


def tune_instant_ramen(X, Y, T, n_features, n_env, use_xgboost, logger):
    results = []

    for lr, init_temp, anneal_rate in PARAM_COMBINATIONS:
        logger.info("Testing with lr=%s, init_temp=%s, anneal_rate=%s", lr, init_temp, anneal_rate)

        instant_ramen = InstantRamen(
            input_dim=n_features,
            n_env=n_env,
            init_temp=init_temp,
            learning_rate=lr,
            anneal_rate=anneal_rate,
            use_xgboost=use_xgboost,
            logger=logger,
        )

        subset, loss = instant_ramen.compute_subset(X, Y, T)

        results.append({
            'learning_rate': lr,
            'init_temp': init_temp,
            'anneal_rate': anneal_rate,
            'loss': loss,
            'subset': subset
        })

    best_result = min(results, key=lambda x: x['loss'])
    log_best_params(best_result, logger)

    return best_result['subset']


def evaluate_naive_baselines(data, use_xgboost):
    X, T, Y = pool_data(data, 'X'), pool_data(data, 'T'), pool_data(data, 'Y')
    n_env = len(data)

    X_pool = X.reshape(-1, X.shape[2])
    T_pool = T.reshape(-1)
    Y_pool = Y.reshape(-1)

    # Model selection
    if use_xgboost:
        mu_t_all = XGBClassifier(**XGB_PARAMS)
        mu_y0_all = XGBRegressor(**XGB_PARAMS)
        mu_y1_all = XGBRegressor(**XGB_PARAMS)
    else:
        mu_t_all = LogisticRegression()
        mu_y0_all = LinearRegression()
        mu_y1_all = LinearRegression()

    # Model fitting
    mu_t_all.fit(X_pool, T_pool)
    mu_y0_all.fit(X_pool[T_pool == 0], Y_pool[T_pool == 0])
    mu_y1_all.fit(X_pool[T_pool == 1], Y_pool[T_pool == 1])

    error_all, error_null = 0, 0

    for e in range(n_env):
        T_e, Y_e = T[e, :], Y[e, :]
        X_e = X[e, :, :].reshape(-1, X.shape[2])

        # Null model error
        ate_null = (Y_e * T_e) / (T_e == 1).mean() - (Y_e * (1 - T_e)) / (T_e == 0).mean()
        error_null += np.abs(data[e]['ATE'] - ate_null.mean())

        # Baseline model error
        pi1_all = np.clip(mu_t_all.predict_proba(X_e)[:, 1], EPSILON, 1 - EPSILON)
        pi0_all = 1 - pi1_all

        ate_all = (
            mu_y1_all.predict(X_e) - mu_y0_all.predict(X_e)
            + T_e / pi1_all * (Y_e - mu_y1_all.predict(X_e))
            - (1 - T_e) / pi0_all * (Y_e - mu_y0_all.predict(X_e))
        )
        error_all += np.abs(data[e]['ATE'] - ate_all.mean())

    return error_all / n_env, error_null / n_env


def evaluate_subset(subset, data, use_xgboost=False):
    X, T, Y = pool_data(data, 'X'), pool_data(data, 'T'), pool_data(data, 'Y')
    n_env = len(data)

    X_pool = X[:, :, subset].reshape(-1, len(subset))
    T_pool, Y_pool = T.reshape(-1), Y.reshape(-1)

    # Model selection
    if use_xgboost:
        mu_t = XGBClassifier(**XGB_PARAMS)
        mu_y0 = XGBRegressor(**XGB_PARAMS)
        mu_y1 = XGBRegressor(**XGB_PARAMS)
    else:
        mu_t = LogisticRegression()
        mu_y0 = LinearRegression()
        mu_y1 = LinearRegression()

    # Model fitting
    mu_t.fit(X_pool, T_pool)
    mu_y0.fit(X_pool[T_pool == 0], Y_pool[T_pool == 0])
    mu_y1.fit(X_pool[T_pool == 1], Y_pool[T_pool == 1])

    error = 0
    for e in range(n_env):
        T_e, Y_e = T[e, :], Y[e, :]
        X_e = X[e, :, subset].T

        pi1 = np.clip(mu_t.predict_proba(X_e)[:, 1], EPSILON, 1 - EPSILON)
        pi0 = 1 - pi1

        hat_ate = (
            mu_y1.predict(X_e) - mu_y0.predict(X_e)
            + T_e / pi1 * (Y_e - mu_y1.predict(X_e))
            - (1 - T_e) / pi0 * (Y_e - mu_y0.predict(X_e))
        )
        error += np.abs(data[e]['ATE'] - hat_ate.mean())

    return error / n_env


def evaluate_irm(data):
    X, T, Y = pool_data(data, 'X'), pool_data(data, 'T'), pool_data(data, 'Y')
    n_env = len(data)

    environments = []
    for e in range(n_env):
        X_e = np.hstack((np.ones((X[e].shape[0], 1)), X[e]))
        T_e = T[e, :].reshape(-1, 1)
        Y_e = Y[e, :].reshape(-1, 1)
        environments.append((torch.Tensor(X_e), torch.Tensor(T_e), 0, torch.Tensor(Y_e)))

    irm = IRM(environments, {})
    irm_ates = irm.ate(n_env)

    error = sum(np.abs(irm_ates[e].detach().numpy() - data[e]['ATE']) for e in range(n_env))
    return error / n_env
