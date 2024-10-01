from functools import partial

import jax
from jax import jit, vmap, numpy as jnp
from jax.scipy.special import logsumexp


def compute_inv_chol_S(chol_S):
    D = chol_S.shape[-1]
    identity = jnp.eye(D)
    inv_chol = jax.scipy.linalg.solve_triangular(chol_S, identity, lower=True)
    inv_S = inv_chol @ inv_chol.T
    return inv_S


def compute_responsibilities(y_t, x_t_k, P_t_k, pi_t, R_t_k, H):
    B, N, D = y_t.shape
    _, K, D_s = x_t_k.shape

    # Compute mean and covariance for each component
    mean_k = jnp.einsum("bkd,sd->bks", x_t_k, H.T)  # [B, K, D]

    # Compute S_t_k = H @ P_t_k @ H.T + R_t_k
    S_t_k = jnp.einsum("bksd,sd->bks", jnp.einsum("sd,bkds->bksd", H, P_t_k), H.T) + R_t_k  # [B, K, D, D]
    S_t_k += 1e-6 * jnp.eye(D)  # Ensure positive definiteness

    # Precompute inverses and determinants
    chol_S_t_k = vmap(vmap(jnp.linalg.cholesky))(S_t_k)  # [B, K, D, D]
    inv_S_t_k = vmap(vmap(compute_inv_chol_S))(chol_S_t_k)  # [B, K, D, D]
    log_det_S_t_k = 2 * jnp.sum(jnp.log(jnp.diagonal(chol_S_t_k, axis1=-2, axis2=-1) + 1e-6), axis=-1)  # [B, K]

    # Compute log-likelihoods
    def log_likelihood(y_n, mean_k, inv_S_k, log_det_S_k):
        diff = y_n[:, None, :] - mean_k  # [B, K, D]
        mahalanobis = -0.5 * jnp.sum(diff[..., None] * (inv_S_k @ diff[..., None]), axis=(-2, -1))  # [B, K]
        return mahalanobis - 0.5 * log_det_S_k - 0.5 * D * jnp.log(2 * jnp.pi)

    log_likelihoods = vmap(log_likelihood, in_axes=(0, 0, 0, 0))(y_t, mean_k, inv_S_t_k, log_det_S_t_k)  # [B, N, K]

    # Compute responsibilities
    log_pi_t = jnp.log(pi_t + 1e-6)  # [B, K]
    log_gamma_numerators = log_pi_t[:, None, :] + log_likelihoods  # [B, N, K]
    log_gamma_denominator = logsumexp(log_gamma_numerators, axis=-1, keepdims=True)  # [B, N, 1]
    gamma_nk = jnp.exp(log_gamma_numerators - log_gamma_denominator)  # [B, N, K]
    return gamma_nk


def update_mixture_coefficients(gamma_nk):
    N = gamma_nk.shape[1]
    pi_t = jnp.sum(gamma_nk, axis=1) / N  # [B, K]
    return pi_t


def update_observation_noise_covariance(y_t, x_t_k, gamma_nk, H):
    B, N, D = y_t.shape
    _, K, D_s = x_t_k.shape

    mean_k = jnp.einsum("bkd,sd->bks", x_t_k, H.T)  # [B, K, D]
    diff = y_t[:, :, None, :] - mean_k[:, None, :, :]  # [B, N, K, D]
    weighted_diff_sq = gamma_nk[:, :, :, None] * diff**2  # [B, N, K, D]
    numerator = jnp.sum(weighted_diff_sq, axis=1)  # [B, K, D]
    denominator = jnp.sum(gamma_nk, axis=1)[:, :, None] + 1e-6  # [B, K, 1]

    R_diag = numerator / denominator  # [B, K, D]
    R_t_k = vmap(lambda batch_diag: vmap(jnp.diag)(batch_diag))(R_diag)  # [B, K, D, D]
    R_t_k += 1e-6 * jnp.eye(D)  # Ensure positive definiteness
    return R_t_k


def kalman_predict(x_t_prev_k, P_t_prev_k, A, Q):
    x_pred_k = jnp.einsum("ij,bkj->bki", A, x_t_prev_k)  # [B, K, D_s]
    P_pred_k = jnp.einsum("ij,bkjl,lk->bkil", A, P_t_prev_k, A.T) + Q  # [B, K, D_s, D_s]
    return x_pred_k, P_pred_k


def compute_effective_measurement(y_t, gamma_nk):
    numerator = jnp.einsum("bnk,bnd->bkd", gamma_nk, y_t)  # [B, K, D]
    denominator = jnp.sum(gamma_nk, axis=1)[:, :, None] + 1e-6  # [B, K, 1]
    y_tilde_k = numerator / denominator  # [B, K, D]
    return y_tilde_k


def compute_adjusted_covariance(R_t_k, gamma_nk):
    denominator = jnp.sum(gamma_nk, axis=1)[:, :, None, None] + 1e-6  # [B, K, 1, 1]
    R_tilde_k = R_t_k / denominator  # [B, K, D, D]
    R_tilde_k += 1e-6 * jnp.eye(R_t_k.shape[-1])  # Ensure positive definiteness
    return R_tilde_k


def kalman_update(x_pred_k, P_pred_k, y_tilde_k, R_tilde_k, H):
    # Compute S_t_k = H @ P_pred_k @ H.T + R_tilde_k
    S_t_k = jnp.einsum("bkjl,sl->bkj", jnp.einsum("sl,bkjl->bkjl", H, P_pred_k), H.T) + R_tilde_k  # [B, K, D, D]
    S_t_k += 1e-6 * jnp.eye(S_t_k.shape[-1])  # Ensure positive definiteness

    # Kalman Gain: K_t_k = P_pred_k @ H.T @ inv(S_t_k)
    def compute_gain(P_pred, S):
        K_gain = P_pred @ H.T @ jnp.linalg.inv(S)
        return K_gain

    K_t_k = vmap(vmap(compute_gain))(P_pred_k, S_t_k)  # [B, K, D_s, D]

    # Innovation: y_tilde_k - H @ x_pred_k
    innovation = y_tilde_k - jnp.einsum("bkj,ij->bki", x_pred_k, H.T)  # [B, K, D]

    # Update state estimate
    x_updated = x_pred_k + jnp.einsum("bkij,bkj->bki", K_t_k, innovation)  # [B, K, D_s]

    # Update covariance estimate
    I = jnp.eye(x_updated.shape[-1])
    P_updated = (I - jnp.einsum("bksl,sl->bksl", K_t_k, H)) @ P_pred_k  # [B, K, D_s, D_s]

    return x_updated, P_updated


def run_kalman_filters(y_data, K, H, A, Q, x_init_k, P_init_k, pi_init, R_init_k):
    B, T, N, D = y_data.shape
    D_s = x_init_k.shape[-1]
    x_t_k = x_init_k  # [B, K, D_s]
    P_t_k = P_init_k  # [B, K, D_s, D_s]
    pi_t = pi_init  # [B, K]
    R_t_k = R_init_k  # [B, K, D, D]

    # Store results for plotting
    x_estimates = []
    pi_estimates = []

    for t in range(T):
        y_t = y_data[:, t, :, :]  # [B, N, D]
        # E-Step
        gamma_nk = compute_responsibilities(y_t, x_t_k, P_t_k, pi_t, R_t_k, H)
        # M-Step
        pi_t = update_mixture_coefficients(gamma_nk)
        R_t_k = update_observation_noise_covariance(y_t, x_t_k, gamma_nk, H)
        # Kalman Predict
        x_pred_k, P_pred_k = kalman_predict(x_t_k, P_t_k, A, Q)
        # Effective Measurement
        y_tilde_k = compute_effective_measurement(y_t, gamma_nk)
        R_tilde_k = compute_adjusted_covariance(R_t_k, gamma_nk)
        # Kalman Update
        x_t_k, P_t_k = kalman_update(x_pred_k, P_pred_k, y_tilde_k, R_tilde_k, H)

        # Store estimates
        x_estimates.append(x_t_k)
        pi_estimates.append(pi_t)

    x_estimates = jnp.stack(x_estimates, axis=1)  # [B, T, K, D_s]
    pi_estimates = jnp.stack(pi_estimates, axis=1)  # [B, T, K]
    return x_estimates, pi_estimates
