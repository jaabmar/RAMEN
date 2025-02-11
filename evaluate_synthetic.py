import argparse
import logging
import os
import sys

from data import sample_synthetic
from models.ramen import Ramen
from utils import (evaluate_irm, evaluate_naive_baselines, evaluate_subset,
                   pool_data, tune_instant_ramen)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Synthetic Experiments")
    parser.add_argument("--n_env", type=int, default=5, help="Number of environments")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples for each environment")
    parser.add_argument("--n_features", type=int, default=2, help="Dimension of X")
    parser.add_argument("--invariance", type=str, default='Y')
    parser.add_argument("--post_treatment", type=str, default='collider')
    parser.add_argument("--n_post", type=int, default=1, help="Number of post-treatment features")
    parser.add_argument("--ate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logger(post_treatment, invariance, seed):
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/synthetic_{post_treatment}_{invariance}_seed{seed}.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main():
    args = parse_arguments()
    logger = setup_logger(args.post_treatment, args.invariance, args.seed)

    data = sample_synthetic(
        n_env=args.n_env, n_samples=args.n,
        n_features=args.n_features, invariance=args.invariance,
        post_treatment_type=args.post_treatment,
        n_post_treatment=args.n_post, ate=args.ate, seed=args.seed
    )
    X, T, Y = map(lambda var: pool_data(data, var), ['X', 'T', 'Y'])
    obs_features = X.shape[-1]

    error_all, error_null = evaluate_naive_baselines(data, use_xgboost=False)

    ramen = Ramen(obs_features, args.n_env, use_xgboost=False, logger=logger)
    subset_ramen = ramen.compute_subset(X, Y, T)
    error_ramen = evaluate_subset(subset_ramen, data, use_xgboost=False)

    logger.info("Ramen Subset: [%s], MAE: %.3f", ', '.join(map(str, subset_ramen)), error_ramen)

    subset_instant_ramen = tune_instant_ramen(X, Y, T, obs_features, args.n_env, use_xgboost=False, logger=logger)
    error_instant_ramen = evaluate_subset(subset_instant_ramen, data, use_xgboost=False)

    logger.info("Instant Ramen Subset: [%s], MAE: %.3f", ', '.join(map(str, subset_instant_ramen)), error_instant_ramen)

    error_irm = evaluate_irm(data)

    logger.info(
        "MAE (null): %.3f, MAE (all): %.3f, MAE (Ramen): %.3f, MAE (Instant Ramen): %.3f, MAE (IRM): %.3f",
        error_null, error_all, error_ramen, error_instant_ramen, error_irm
    )


if __name__ == "__main__":
    main()
