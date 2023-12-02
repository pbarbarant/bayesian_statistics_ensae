# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
from sklearn import linear_model
from scipy import special
from numba import njit, jit

plt.style.use("fivethirtyeight")
PARALLEL = False

# deactivate numba for debugging
# njit = lambda x: x
# jit = lambda x: x


def sample_one_dataset(
    s,
    R_y,
    k=100,
    T=200,
    rho=0.75,
    a=1,
    b=1,
    A=1,
    B=1,
):
    X = np.zeros((T, k))
    toeplitz_corr = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            toeplitz_corr[i, j] = rho ** np.abs(i - j)

    for t in range(T):
        X[t, :] = np.random.multivariate_normal(np.zeros(k), toeplitz_corr)

    # Initialize R2 using beta distribution
    R2 = stats.beta(A, B).rvs()

    # Initialize q using beta distribution
    q = stats.beta(a, b).rvs()

    # Initialize z as a random vector of s ones and k-s zeros
    z = np.array([0.0] * (k - s) + [1.0] * s)
    np.random.shuffle(z)

    # Initialize beta as a random vector of k values
    beta = (np.random.randn(k) * z).reshape(-1, 1)

    sigma2 = (1 / R_y - 1) / T * np.sum((X @ beta) ** 2)
    eps = np.random.multivariate_normal(np.zeros(T), sigma2 * np.eye(T)).reshape(-1, 1)

    # Standardize X and eps
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    eps = (eps - np.mean(eps)) / np.std(eps)

    # Sanity checks
    assert X.shape == (T, k)
    assert eps.shape == (T, 1)
    assert beta.shape == (k, 1)
    assert z.shape == (k,)
    assert R2 > 0 and R2 < 1
    assert q > 0 and q < 1
    assert sigma2 > 0

    return X, beta, eps, sigma2, R2, q, z


@njit
def posterior_R2_q(R2, q, beta, vx, sigma2, z, a=1, b=1, A=1, B=1, k=100):
    s = int(np.sum(z))
    return (
        np.exp(
            -1
            / (2 * sigma2)
            * (k * vx * q * (1 - R2))
            / (R2)
            * (beta.T @ np.diag(z) @ beta)
        )
        * q ** (s + s / 2 + a - 1)
        * (1 - q) ** (k - s + b - 1)
        * R2 ** (A - 1 - s / 2)
        * (1 - R2) ** (s / 2 + B - 1)
    )


@jit
def sample_joint_R2_q(X, z, beta, sigma2):
    # Create grid of R2 and q values
    x = np.concatenate(
        [
            np.arange(0.001, 0.1, 0.001),
            np.arange(0.1, 0.9, 0.01),
            np.arange(0.9, 0.999, 0.001),
        ]
    )
    Rs, qs = np.meshgrid(x, x)
    vx = compute_vx(X)

    # Compute posterior
    posterior = posterior_R2_q(Rs, qs, beta, vx, sigma2, z)
    posterior /= np.sum(posterior)

    # Plot posterior
    # plt.contourf(Rs, qs, posterior)
    # plt.show()

    # Sample R2 and q
    R2 = np.random.choice(Rs.flatten(), p=posterior.flatten())
    q = np.random.choice(qs.flatten(), p=posterior.flatten())

    return R2, q


@njit
def compute_prob_z_i(X, eps, beta, z, q, gamma, i, T=200):
    Y = (X @ beta) + eps

    # Compute W_tilde_0 and W_tilde_1 depending on z_i
    z[i] = 0
    X_tilde_0 = X[:, z == 1]
    W_tilde_0 = X_tilde_0.T @ X_tilde_0 + np.eye(int(np.sum(z))) / gamma**2
    z[i] = 1
    X_tilde_1 = X[:, z == 1]
    W_tilde_1 = X_tilde_1.T @ X_tilde_1 + np.eye(int(np.sum(z))) / gamma**2

    # Fast computation of beta_tilde_0 and beta_tilde_1
    beta_tilde_0 = np.linalg.solve(W_tilde_0, X_tilde_0.T @ Y)
    beta_tilde_1 = np.linalg.solve(W_tilde_1, X_tilde_1.T @ Y)

    Y_tilde_0 = (X_tilde_0 @ beta_tilde_0) + eps
    Y_tilde_1 = (X_tilde_1 @ beta_tilde_1) + eps

    # Fast computation of the log-determinant of W_tilde_0 and W_tilde_1
    log_det_W_tilde_0 = np.trace(np.log(W_tilde_0))
    log_det_W_tilde_1 = np.trace(np.log(W_tilde_1))

    # Compute the log of the probability ratio
    log_ratio = (
        np.log(gamma)
        + np.log(1 - q)
        - np.log(q)
        - 1 / 2 * log_det_W_tilde_0
        + 1 / 2 * log_det_W_tilde_1
        - T
        / 2
        * np.log((Y_tilde_0.T @ Y_tilde_0 - beta_tilde_0.T @ W_tilde_0 @ beta_tilde_0))
        + T
        / 2
        * np.log((Y_tilde_1.T @ Y_tilde_1 - beta_tilde_1.T @ W_tilde_1 @ beta_tilde_1))
    )

    # Compute the probability of z_i = 1
    ratio = np.exp(log_ratio)
    prob = ratio / (1 + ratio)

    return prob


@njit
def sample_z(X, eps, beta, z, R2, q, T=200):
    """Sample z using one gibbs iteration"""
    gamma = np.sqrt(compute_gamma2(X, R2, q))
    for i in range(z.shape[0]):
        prob = compute_prob_z_i(X, eps, beta, z, q, gamma, i, T)
        # Sample z_i
        if np.random.rand() < prob:
            z[i] = 1
        else:
            z[i] = 0
    return z


@njit
def compute_vx(X):
    """Compute the mean of the variance of X"""
    _, k = X.shape
    vx = 0
    for j in range(k):
        vx += np.var(X[:, j])
    vx /= k
    return vx


@njit
def compute_gamma2(X, R2, q, k=100):
    """Compute gamma^2 using the formula given in the assignment"""
    vx = compute_vx(X)
    return R2 / ((1 - R2) * k * q * vx)


@njit
def compute_sigma2_scale(X, eps, beta, R2, q, z, T=200):
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    beta_tilde = beta[z == 1]
    Y_tilde = (X_tilde @ beta_tilde) + eps
    gamma2 = compute_gamma2(X, R2, q)
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / gamma2
    beta_tilde_hat = np.linalg.inv(W_tilde) @ X_tilde.T @ Y_tilde
    scale = (Y_tilde.T @ Y_tilde - beta_tilde_hat.T @ W_tilde @ beta_tilde_hat) / 2
    return scale.item()


@jit
def sample_sigma2(X, eps, beta, R2, q, z, T=200):
    """Sample sigma2 using formula 4"""
    scale = compute_sigma2_scale(X, eps, beta, R2, q, z)
    return stats.invgamma(T / 2, scale=scale).rvs()


@njit
def compute_beta_tilde_param(X, eps, beta, R2, q, sigma2, z):
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    beta_tilde = beta[z == 1]
    Y_tilde = (X_tilde @ beta_tilde) + eps
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / compute_gamma2(X, R2, q)
    W_tilde_inv = np.linalg.inv(W_tilde)
    beta_tilde_hat = W_tilde_inv @ X_tilde.T @ Y_tilde
    return beta_tilde_hat, W_tilde_inv


@jit
def sample_beta_tilde(X, eps, beta, R2, q, sigma2, z):
    """Sample sigma2 using formula 5"""
    beta_tilde_hat, W_tilde_inv = compute_beta_tilde_param(
        X, eps, beta, R2, q, sigma2, z
    )
    return np.random.multivariate_normal(
        beta_tilde_hat.reshape(-1), sigma2 * W_tilde_inv
    ).reshape(-1, 1)


@jit
def sample_beta(X, eps, beta, R2, q, sigma2, z, k=100):
    """Auxiliary function to sample beta from beta_tilde"""
    beta_tilde = sample_beta_tilde(X, eps, beta, R2, q, sigma2, z)
    beta = np.zeros((k, 1))
    beta[z == 1] = beta_tilde
    return beta.reshape(-1, 1)


@jit
def one_gibbs_iteration(X, eps, R2, q, z, sigma2, beta):
    """Run one iteration of the Gibbs sampler"""
    R2, q = sample_joint_R2_q(X, z, beta, sigma2)
    # z = sample_z(X, eps, beta, z, R2, q)
    sigma2 = sample_sigma2(X, eps, beta, R2, q, z)
    beta = sample_beta(X, eps, beta, R2, q, sigma2, z)
    # print(f"R2={R2:.3f}, q={q:.3f}, sigma2={sigma2:.3f}")
    return R2, q, z, sigma2, beta


def make_plots(q, medians, s, R_y):
    """Plot posterior median of q and marginal posterior distribution of q for a given dataset"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Add histograms
    ax1.hist(medians, bins=50, density=True)
    ax2.hist(q, bins=50, density=True)

    # Add kernel density estimations
    kde1 = stats.gaussian_kde(medians)
    x = np.linspace(medians.min(), medians.max(), 1000)
    ax1.plot(x, kde1(x), label="KDE")
    kde2 = stats.gaussian_kde(q)
    y = np.linspace(q.min(), q.max(), 1000)
    ax2.plot(y, kde2(y), label="KDE")

    # Add labels and titles
    ax1.set_xlabel("Posterior median of q")
    ax1.set_ylabel("Density")
    ax1.set_title(
        f"Histogram of posterior median of q\nwith s={s} and R_y={int(R_y*100)}%"
    )
    ax1.legend()
    ax2.set_xlabel("q")
    ax2.set_ylabel("Density")
    ax2.set_title(
        f"Histogram of marginal posterior distribution of q\nwith s={s} and R_y={int(R_y*100)}% (last dataset)"
    )
    ax2.legend()
    # plt.show()
    fig.savefig(f"figures/s={s}_R_y={int(R_y*100)}.png")


if __name__ == "__main__":
    N_datasets = 3
    N_iter = 10000
    burn_in = 1000

    for s, R_y in product([5, 10, 100], [0.02, 0.25, 0.5]):
        q_matrix = np.zeros((N_datasets, N_iter))
        for i in range(N_datasets):
            print(f"Dataset {i}, s={s}, R_y={R_y}")
            X, beta, eps, sigma2, R2, q, z = sample_one_dataset(s, R_y)
            for k in tqdm(range(N_iter)):
                R2, q, z, sigma2, beta = one_gibbs_iteration(
                    X, eps, R_y, q, z, sigma2, beta
                )
                q_matrix[i, k] = q

        q_matrix = q_matrix[:, burn_in:]
        posterior_median_q = np.median(q_matrix, axis=1)
        # Plot posterior median of q and marginal posterior distribution of q for the last dataset
        make_plots(q_matrix[-1, :], posterior_median_q, s, R_y)
