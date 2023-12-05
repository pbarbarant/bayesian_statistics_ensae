# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats

# Use numba for just-in-time compilation
from numba import njit

# Use submitit to run jobs on slurm clusters
import submitit

log_folder = "logs/%j"
executor = submitit.AutoExecutor(folder=log_folder)

# Set to True to run the computation in parallel
PARALLEL = True
# Set to True to run the computation on a slurm cluster
CLUSTER = True


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
    '''Sample one dataset from the model'''
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
    z = np.array([0] * (k - s) + [1] * s)
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
def posterior_R2_q(R2, q, sigma2, beta_tilde_norm2, s, a=1, b=1, A=1, B=1):
    """Auxiliary function to compute the posterior on a grid of R2 and q values"""
    vx = 1  # X is standardized
    k = 100
    # Compute the log of the posterior
    log_posterior = (
        -1 / (2 * sigma2) * (k * vx * q * (1 - R2)) / (R2) * beta_tilde_norm2
        + (s + s / 2 + a - 1) * np.log(q)
        + (k - s + b - 1) * np.log(1 - q)
        + (A - 1 - s / 2) * np.log(R2)
        + (B - 1 + s / 2) * np.log(1 - R2)
    )
    return np.exp(log_posterior)


@njit
def compute_posterior_grid(Rs, qs, z, beta, sigma2):
    """Compute the posterior on a grid of R2 and q values"""
    s = int(np.sum(z))
    # vx = np.mean(np.var(X, axis=0))

    beta_tilde = beta[z == 1]
    beta_tilde_norm = beta_tilde.T @ beta_tilde
    # Compute posterior
    posterior = posterior_R2_q(Rs, qs, sigma2, beta_tilde_norm)
    # Normalize posterior
    posterior = posterior / np.sum(posterior)
    return posterior


def sample_joint_R2_q(Rs, qs, z, beta, sigma2):
    posterior = compute_posterior_grid(Rs, qs, z, beta, sigma2)

    # Sample R2 and q
    R2 = np.random.choice(Rs.flatten(), p=posterior.flatten())
    q = np.random.choice(qs.flatten(), p=posterior.flatten())

    return R2, q


@njit
def sample_z(X, eps, beta, z, R2, q):
    """Sample z using one gibbs iteration"""
    T = 200
    gamma = np.sqrt(compute_gamma2(R2, q))
    Y = (X @ beta) + eps
    for i in range(z.shape[0]):
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

        # # Compute the log of the probability ratio
        log_ratio = (
            np.log(gamma)
            + np.log(1 - q)
            - np.log(q)
            - 1 / 2 * log_det_W_tilde_0
            + 1 / 2 * log_det_W_tilde_1
            - T
            / 2
            * np.log(
                (Y_tilde_0.T @ Y_tilde_0 - beta_tilde_0.T @ W_tilde_0 @ beta_tilde_0)
            )
            + T
            / 2
            * np.log(
                (Y_tilde_1.T @ Y_tilde_1 - beta_tilde_1.T @ W_tilde_1 @ beta_tilde_1)
            )
        ).item()
        # Compute the ratio
        # ratio = (
        #     gamma
        #     * (1 - q)
        #     / (q)
        #     * np.sqrt(np.linalg.det(W_tilde_1) / np.linalg.det(W_tilde_0)) ** (-1/2)
        #     * (Y_tilde_0.T @ Y_tilde_0 - beta_tilde_0.T @ W_tilde_0 @ beta_tilde_0) ** (
        #         -T / 2
        #     )
        #     * (Y_tilde_1.T @ Y_tilde_1 - beta_tilde_1.T @ W_tilde_1 @ beta_tilde_1) ** (
        #         T / 2
        #     )
        # ).item()

        # Compute the probability of z_i = 1
        ratio = np.exp(log_ratio)
        prob = 1 / (1 + np.exp(log_ratio))

        # Sample z_i
        if np.random.rand() < prob:
            z[i] = 1
        else:
            z[i] = 0
    return z


@njit
def compute_gamma2(R2, q):
    """Compute gamma^2 using the formula given in the assignment"""
    # vx = np.mean(np.var(X, axis=1))
    vx = 1
    k = 100
    return R2 / ((1 - R2) * k * q * vx)


@njit
def sample_sigma2_scale(X, eps, beta, R2, q, z):
    """Sample sigma2 using formula 4"""
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    beta_tilde = beta[z == 1]
    Y_tilde = (X_tilde @ beta_tilde) + eps
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / compute_gamma2(R2, q)
    # Fast computation of beta_tilde_hat
    beta_tilde_hat = np.linalg.solve(W_tilde, X_tilde.T @ Y_tilde)
    scale = (Y_tilde.T @ Y_tilde - beta_tilde_hat.T @ W_tilde @ beta_tilde_hat) / 2
    return scale


def sample_sigma2(X, eps, beta, R2, q, z):
    """Sample sigma2 using formula 4"""
    T = 200
    scale = sample_sigma2_scale(X, eps, beta, R2, q, z)
    return stats.invgamma(T / 2, scale=scale).rvs()


@njit
def sample_beta_tilde_scale(X, eps, beta, R2, q, sigma2, z):
    """Sample sigma2 using formula 5"""
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    beta_tilde = beta[z == 1]
    Y_tilde = (X_tilde @ beta_tilde) + eps
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / compute_gamma2(R2, q)
    W_tilde_inv = np.linalg.inv(W_tilde)
    beta_tilde_hat = W_tilde_inv @ X_tilde.T @ Y_tilde
    mean = beta_tilde_hat.reshape(-1)
    cov = sigma2 * W_tilde_inv
    return mean, cov


def sample_beta_tilde(X, eps, beta, R2, q, sigma2, z):
    """Sample sigma2 using formula 5"""
    mean, cov = sample_beta_tilde_scale(X, eps, beta, R2, q, sigma2, z)
    return np.random.multivariate_normal(mean, cov).reshape(-1, 1)


def sample_beta(X, eps, beta, R2, q, sigma2, z):
    """Auxiliary function to sample beta from beta_tilde"""
    beta_tilde = sample_beta_tilde(X, eps, beta, R2, q, sigma2, z)
    beta = np.zeros((100, 1))
    beta[z == 1] = beta_tilde
    return beta.reshape(-1, 1)


def one_gibbs_iteration(X, eps, R2, q, z, sigma2, beta, Rs, qs):
    """Run one iteration of the Gibbs sampler"""
    R2, q = sample_joint_R2_q(Rs, qs, z, beta, sigma2)
    sampled_z = sample_z(X, eps, beta, z, R2, q)
    # If sampled_z is a vector of zeros, we keep the previous value of z
    if sampled_z.sum() == 0:
        return R2, q, z, sigma2, beta
    else:
        z = sampled_z
    sigma2 = sample_sigma2(X, eps, beta, R2, q, z)
    beta = sample_beta(X, eps, beta, R2, q, sigma2, z)
    # print(f"R2={R2:.3f}, q={q:.3f}, sigma2={sigma2:.3f}")
    return R2, q, z, sigma2, beta


def compute_one_dataset(R_y, s, N_iter):
    """Sample one dataset and run the Gibbs sampler for N_iter iterations"""
    X, beta, eps, sigma2, R2, q, z = sample_one_dataset(s, R_y)
    q_chain = np.zeros(N_iter)
    # Create grid of R2 and q values
    x = np.concatenate(
        [
            np.arange(0.001, 0.1, 0.001),
            np.arange(0.1, 0.9, 0.01),
            np.arange(0.9, 0.999, 0.001),
        ]
    )
    Rs, qs = np.meshgrid(x, x)
    # Run Gibbs sampler
    for k in range(N_iter):
        R2, q, z, sigma2, beta = one_gibbs_iteration(
            X, eps, R2, q, z, sigma2, beta, Rs, qs
        )
        q_chain[k] = q
    return q_chain


def job_wrapper(s, R_y, N_iter, burn_in):
    q_matrix = np.zeros((N_datasets, N_iter))
    if PARALLEL:
        results = Parallel(n_jobs=-1)(
            delayed(compute_one_dataset)(R_y, s, N_iter)
            for _ in tqdm(range(N_datasets))
        )
        for i, q_chain in enumerate(results):
            q_matrix[i, :] = q_chain
    else:
        for i in range(N_datasets):
            q_matrix[i, :] = compute_one_dataset(R_y, s, N_iter)
    q_matrix = q_matrix[:, burn_in:]
    posterior_median_q = np.median(q_matrix, axis=1)
    # Save q_matrix
    np.save(f"q_matrix/s={s}_R_y={int(R_y*100)}.npy", q_matrix)
    # Plot posterior median of q and marginal posterior distribution of q for the last dataset
    make_plots(q_matrix[-1, :], posterior_median_q, s, R_y)


def make_plots(q, medians, s, R_y):
    """Plot posterior median of q and marginal posterior distribution of q for a given dataset"""
    plt.style.use("fivethirtyeight")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Add histograms
    ax1.hist(np.round(medians, decimals=3), bins=100, density=True)
    ax2.hist(q, bins=100, density=True)

    # Add kernel density estimations for q marginal posterior distribution
    kde = stats.gaussian_kde(q)
    x = np.linspace(q.min(), q.max(), 1000)
    ax2.plot(x, kde(x), label="KDE")

    # Add vertical lines for the median of q on ax2
    ax2.axvline(np.median(q), color="black", label="Empirical median")

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
    N_datasets = 100
    N_iter = 110_000
    burn_in = 10_000

    list_s = [5, 10, 100]
    list_R_y = [0.02, 0.25, 0.5]

    if CLUSTER:
        # Create product of list_s and list_R_y
        product_s_R_y = np.array(list(product(list_s, list_R_y)))
        executor = submitit.AutoExecutor(folder=log_folder)
        executor.update_parameters(
            # cpus_per_task=N_datasets,
            slurm_partition="normal",
            slurm_job_name="gibbs",
            slurm_time="50:00:00",
        )
        jobs = executor.map_array(
            job_wrapper,
            product_s_R_y[:, 0].astype(int),
            product_s_R_y[:, 1],
            [N_iter] * len(product_s_R_y),
            [burn_in] * len(product_s_R_y),
        )

    else:
        for s, R_y in product(list_s, list_R_y):
            print(f"Running simulation with s={s} and R_y={int(R_y*100)}%")
            job_wrapper(s, R_y, N_iter, burn_in)
