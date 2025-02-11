import numpy as np
import pandas as pd
import torch
import os


def sample_synthetic(n_env, n_samples, n_features=10, invariance='T', post_treatment_type='collider',
                     n_post_treatment=1, ate=1.0, seed=1):
    np.random.seed(seed)
    data = []
    beta_xt = np.random.normal(size=(n_features))
    beta_xt /= np.linalg.norm(beta_xt)

    beta_y = np.random.normal(size=n_features)
    beta_y /= np.linalg.norm(beta_y)

    for _ in range(n_env):
        shift = np.random.normal(size=2)
        shift_x = np.random.normal(size=n_features)
        X2 = np.random.normal(size=(n_samples, n_features), loc=shift_x, scale=np.abs(shift_x)**2)
        logit = X2 @ beta_xt

        if invariance == 'T':
            Y = np.random.normal(size=n_samples, loc=shift[0], scale=np.abs(shift[0])**2)
            logit += np.random.normal(size=n_samples)
        elif invariance == 'Y':
            Y = np.random.normal(size=n_samples)
            logit += np.random.normal(size=n_samples, loc=shift[0], scale=np.abs(shift[0])**2)
        else:
            Y = np.random.normal(size=n_samples)
            logit += np.random.normal(size=n_samples)

        p = torch.sigmoid(torch.Tensor(logit))
        T = torch.bernoulli(p).numpy()
        Y += X2 @ beta_y + ate * T

        post_treatment_vars = []
        for _ in range(n_post_treatment):
            if post_treatment_type == 'noise':
                X1 = np.random.normal(loc=shift[1], scale=np.abs(shift[1])**2, size=(n_samples, 1))
            elif post_treatment_type == 'collider':
                X1 = Y.reshape(n_samples, 1) + T.reshape(n_samples, 1) + np.random.normal(loc=shift[1],
                                                                                          scale=np.abs(shift[1])**2,
                                                                                          size=(n_samples, 1))
            else:
                X1 = Y.reshape(n_samples, 1) + np.random.normal(loc=shift[1], scale=np.abs(shift[1])**2, size=(n_samples, 1))
            post_treatment_vars.append(X1)

        post_treatment_vars = np.hstack(post_treatment_vars)

        X = np.hstack((post_treatment_vars, X2))

        environment_data = {
            'X': X,
            'X_inv': X2,
            'X_spu': post_treatment_vars,
            'T': T,
            'Y': Y,
            'ATE': ate
        }
        data.append(environment_data)

    return data


def sample_semisynthetic(n_env, n_features, invariance='T', post_treatment_type="collider",
                         n_post_treatment=1, ate=10.0, difficulty='easy', seed=1):
    np.random.seed(seed)
    data = []

    # Load dataset
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "datasets", "ihdp_obs.csv")
    x = pd.read_csv(file_path, sep='\s+')
    X_columns = [f'X{i + 1}' for i in range(n_features)]
    X_data = x[X_columns].values
    n_samples = len(X_data)

    # Effect assignment
    effect_assignment = [None] * n_features
    confounders = np.random.choice(range(n_features), size=min(4, n_features), replace=False)
    for i in confounders:
        effect_assignment[i] = 'both'
    remaining_indices = [i for i in range(n_features) if i not in confounders]
    for i in remaining_indices:
        effect_assignment[i] = np.random.choice(['Y', 'T'])

    features_affecting_Y = [i for i, effect in enumerate(effect_assignment) if effect in ['Y', 'both']]
    features_affecting_T = [i for i, effect in enumerate(effect_assignment) if effect in ['T', 'both']]

    # Assign coefficients based on difficulty
    if difficulty == 'easy':
        beta_y = np.zeros(n_features)
        beta_t = np.zeros(n_features)

        if features_affecting_Y:
            beta_y[features_affecting_Y] = np.random.normal(size=len(features_affecting_Y))
            beta_y /= np.linalg.norm(beta_y)
        if features_affecting_T:
            beta_t[features_affecting_T] = np.random.normal(size=len(features_affecting_T))
            beta_t /= np.linalg.norm(beta_t)

    elif difficulty == 'hard':
        transformations_y, transformations_t = {}, {}
        beta_y, beta_t = {}, {}

        transformation_choices_y = [
            lambda x: 0.5 * np.log(np.abs(x)),
            lambda x: (x / 2) ** 2,
            lambda x: x + 1,
            lambda x: np.exp(x / 2)
        ]
        transformation_choices_t = [
            lambda x: 0.5 * np.log(np.abs(x)),
            lambda x: (x / 2) ** 2,
            lambda x: x + 0.2,
            lambda x: np.exp(x / 2)
        ]

        for i in range(n_features):
            if effect_assignment[i] in ['Y', 'both']:
                transformations_y[i] = np.random.choice(transformation_choices_y)
                beta_y[i] = np.random.uniform(-2, 2)
            if effect_assignment[i] in ['T', 'both']:
                transformations_t[i] = np.random.choice(transformation_choices_t)
                beta_t[i] = np.random.uniform(-0.5, 0.5)

    else:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Choose 'easy' or 'hard'.")

    # Identify hideable features
    parents_of_Y = {i for i, effect in enumerate(effect_assignment) if effect == 'Y'}
    parents_of_T = {i for i, effect in enumerate(effect_assignment) if effect == 'T'}

    if invariance == 'Y':
        hideable_features = list(parents_of_T)
    elif invariance == 'T':
        hideable_features = list(parents_of_Y)
    else:
        hideable_features = list(parents_of_T | parents_of_Y)

    hidden_feature = np.random.choice(hideable_features, size=1, replace=False) if hideable_features else []

    coef_inv = np.random.uniform(0.5, 1.0)
    coef_x = np.random.uniform(0.5, 1.0)

    # Environment loop
    for e in range(n_env):
        shift_inv = e * coef_inv + np.random.normal(size=1)
        shift_post = e * coef_x + np.random.normal(size=n_post_treatment)
        shift_x = e * coef_x + np.random.normal(size=n_features)

        X2 = X_data + np.random.normal(loc=shift_x, scale=1.0, size=(n_samples, n_features))

        # Outcome and treatment generation
        if difficulty == 'easy':
            logit = X2 @ beta_t if features_affecting_T else np.zeros(n_samples)
            Y = X2 @ beta_y if features_affecting_Y else np.zeros(n_samples)
        else:
            transformed_features_y = [beta_y[i] * transformations_y[i](X2[:, i]) for i in transformations_y]
            transformed_features_t = [beta_t[i] * transformations_t[i](X2[:, i]) for i in transformations_t]
            Y = np.sum(transformed_features_y, axis=0) if transformed_features_y else np.zeros(n_samples)
            logit = np.sum(transformed_features_t, axis=0) if transformed_features_t else np.zeros(n_samples)

        # Invariance adjustments
        if invariance == 'T':
            Y += np.random.normal(loc=shift_inv, scale=1.0, size=n_samples)
            logit += np.random.normal(size=n_samples)
        elif invariance == 'Y':
            logit += np.random.normal(loc=shift_inv / 6, scale=1.0, size=n_samples)
            Y += np.random.normal(size=n_samples)
        elif invariance == "both":
            Y += np.random.normal(size=n_samples)
            logit += np.random.normal(size=n_samples)
        else:
            Y += np.random.normal(loc=shift_inv, scale=1.0, size=n_samples)
            logit += np.random.normal(loc=shift_inv / 6, scale=1.0, size=n_samples)

        # Treatment assignment
        p = torch.sigmoid(torch.Tensor(logit))
        T = torch.bernoulli(p).numpy()
        Y += ate * T

        # Post-treatment variables
        post_treatment_variables = []
        for i in range(n_post_treatment):
            if post_treatment_type == 'collider':
                X1 = Y.reshape(-1, 1) + T.reshape(-1, 1) + np.random.normal(loc=shift_post[i],
                                                                            scale=1.0, size=(n_samples, 1))
            elif post_treatment_type == 'descendant':
                X1 = Y.reshape(-1, 1) + np.random.normal(loc=shift_post[i], scale=1.0, size=(n_samples, 1))
            else:
                X1 = np.random.normal(loc=shift_post[i], scale=1.0, size=(n_samples, 1))
            post_treatment_variables.append(X1)

        post_treatment_variables = np.hstack(post_treatment_variables)
        X2 = np.delete(X2, hidden_feature, axis=1)
        X = np.hstack((post_treatment_variables, X2))

        environment_data = {
            'X': X,
            'post': list(range(n_post_treatment)),
            'confounders': [conf + n_post_treatment for conf in confounders],
            'hidden': [hidden_feature[0] + n_post_treatment] if hidden_feature else [],
            'T': T,
            'Y': Y,
            'ATE': ate,
            'effect_assignment': ["post"] * n_post_treatment + effect_assignment
        }
        data.append(environment_data)

    return data
