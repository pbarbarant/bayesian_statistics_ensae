# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
from scipy import special

plt.style.use("fivethirtyeight")
PARALLEL = False


def sample_one_dataset(k=100, T=200, rho=0.75):
    X = np.zeros((T, k))
    toeplitz_corr = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            toeplitz_corr[i, j] = rho ** np.abs(i - j)

    for t in range(T):
        X[t, :] = np.random.multivariate_normal(np.zeros(k), toeplitz_corr)

    eps = np.random.normal(0, 1, (T, 1))

    return X, eps


def generate_datasets(n_datasets=100, parallel=True):
    if parallel:
        datasets = Parallel(n_jobs=-1)(
            delayed(sample_one_dataset)() for _ in tqdm(range(n_datasets))
        )
    else:
        datasets = []
        for _ in tqdm(range(n_datasets)):
            datasets.append(sample_one_dataset())

    # Concatenate all datasets
    X = np.stack([dataset[0] for dataset in datasets], axis=0)
    eps = np.stack([dataset[1] for dataset in datasets], axis=0)

    # Standardize X and eps
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    eps = (eps - np.mean(eps)) / np.std(eps)

    return X, eps


def sample_joint_R2_q(X, eps, z, beta, R2, q, sigma2, k, T):
    x = np.concatenate([
        np.arange(.001, .1, 0.001),
        np.arange(.1, .9, 0.01),
        np.arange(.9, .999, 0.001)
        ]
    )
    Rs, qs = np.meshgrid(x, x)
    
    def posterior_R2_q(R2, q, a=1, b=1, A=1, B=1):
        s = int(np.sum(z))
        vx = 1/k * np.sum(np.var(X, axis=0))
        return np.exp(-1/(2*sigma2) * (k*vx*q*(1-R2)) / (R2) * (beta.T @ np.diag(z) @ beta).item()) * q**(s+s/2+a-1) * (1-q)**(k-s+b-1) * R2**(A-1-s/2) * (1-R2)**(s/2+B-1)
    
    posterior = posterior_R2_q(Rs, qs)
    # Normalize posterior
    posterior = posterior / np.sum(posterior)
    
    # Sample R2 and q
    R2 = np.random.choice(Rs.flatten(), p=posterior.flatten())
    q = np.random.choice(qs.flatten(), p=posterior.flatten())
    
    return R2, q


def sample_z_i(z, k, q, W_tilde, beta_tilde_hat, Y_tilde, T, gamma):
    s = int(np.sum(z))
    # Problem with prob
    prob = q**s * (1 - q)**(k - s) * (1/gamma**2)**(s/2) * np.sum(np.abs(W_tilde))**(-1/2) * ((Y_tilde.T @ Y_tilde - beta_tilde_hat.T @ W_tilde @ beta_tilde_hat)/2)**(-T/2) * special.gamma(T/2)
    return np.random.binomial(1, prob)


def sample_z(z, k, q, X_tilde, beta_tilde_hat, Y_tilde, T, gamma, n_iter_gibbs=1000):
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / gamma**2
    for _ in range(n_iter_gibbs):
        for i in range(z.shape[0]):
            z[i] = sample_z_i(z, k, q, W_tilde, beta_tilde_hat, Y_tilde, T, gamma)
    return z, int(np.sum(z))


def sample_sigma2(X_tilde, Y_tilde, W_tilde, T):
    beta_tilde_hat = np.linalg.inv(W_tilde) @ X_tilde.T @ Y_tilde
    scale = (Y_tilde.T @ Y_tilde - beta_tilde_hat.T @ W_tilde @ beta_tilde_hat) / 2
    return stats.invgamma(T / 2, scale=scale).rvs()


def sample_beta_tilde(X_tilde, W_tilde, Y_tilde, sigma2):
    W_tilde_inv = np.linalg.inv(W_tilde)
    beta_tilde_hat = W_tilde_inv @ X_tilde.T @ Y_tilde
    return np.random.multivariate_normal(beta_tilde_hat.reshape(-1), sigma2 * W_tilde_inv).reshape(-1, 1)


def update_vectors(X, eps, z, beta_tilde, gamma):
    s = int(np.sum(z))
    X_tilde = X[:, z == 1]
    Y_tilde = (X_tilde @ beta_tilde) + eps
    W_tilde = X_tilde.T @ X_tilde + np.eye(s) / gamma**2
    return X_tilde, Y_tilde, W_tilde


def sample_posterior_marginal_q(X, eps, s, R_y, k=100, T=200, gamma=1e-6):
    q_chain = []
    
    # Initialize values
    R2 = R_y
    q = .1
    z = np.array([0] * (k - s) + [1] * s)
    np.random.shuffle(z)
    beta = (np.random.randn(k) * z).reshape(-1,1)
    sigma2 = (1 / R_y - 1) / T * np.sum((X @ beta)**2)
    
    beta_tilde = beta[z == 1]
    X_tilde, Y_tilde, W_tilde = update_vectors(X, eps, z, beta_tilde, gamma)
    beta_tilde_hat = np.linalg.inv(W_tilde) @ X_tilde.T @ Y_tilde
    
    for iterat in range(110_000):
        print(f"iteration: {iterat}")
        R2, q = sample_joint_R2_q(X, eps, z, beta, R2, q, sigma2, k, T)
        print(f"R2: {R2}, q: {q}")
        # z = sample_z(z, k, q, X_tilde, beta_tilde_hat, Y_tilde, T, gamma)

        X_tilde, Y_tilde, W_tilde = update_vectors(X, eps, z, beta_tilde, gamma)
        
        sigma2 = sample_sigma2(X_tilde, Y_tilde, W_tilde, T)
        
        beta_tilde = sample_beta_tilde(X_tilde, W_tilde, Y_tilde, sigma2)
        
        q_chain.append(q)

    return np.array(q_chain)[10_000:]


def plot_posterior_median_q(medians, s, R_y):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    # Add histogram
    ax.hist(medians, bins=50, density=True)
    # Add kernel density estimation
    kde = stats.gaussian_kde(medians)
    x = np.linspace(medians.min(), medians.max(), 1000)
    ax.plot(x, kde(x), label="KDE")
    ax.set_xlabel("Posterior median of q")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Histogram of posterior median of q with s={s} and R_y={int(R_y*100)}%"
    )
    ax.legend()
    return fig, ax


def plot_marginal_posterior_q(q, s, R_y):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    # Add histogram
    ax.hist(q, bins=50, density=True)
    # Add kernel density estimation
    kde = stats.gaussian_kde(q)
    x = np.linspace(q.min(), q.max(), 1000)
    ax.plot(x, kde(x), label="KDE")
    ax.set_xlabel("q")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Histogram of marginal posterior distribution of q\nwith s={s} and R_y={int(R_y*100)}% (last dataset)"
    )
    ax.legend()
    return fig, ax


def make_plots(q, medians, s, R_y):
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
    plt.show()


if __name__ == "__main__":
    # Question 1
    print("Generating datasets...")
    X, eps = generate_datasets(n_datasets=100)
    print("Done!")

    # Question 2
    for s, R_y in product([5, 10, 100], [0.02, 0.25, 0.5]):
        posterior_median_q = np.zeros(X.shape[0])
        q = np.zeros((X.shape[0], 100_000))

        if PARALLEL:
            q = Parallel(n_jobs=-1)(
                delayed(sample_posterior_marginal_q)(X[i, :], eps[i, :], s, R_y)
                for i in range(X.shape[0])
            )
            q = np.stack(q, axis=0)
        else:
            for i in range(X.shape[0]):
                q[i, :] = sample_posterior_marginal_q(X[i, :], eps[i, :], s, R_y)

        posterior_median_q = np.median(q, axis=1)

        make_plots(q[-1, :], posterior_median_q, s, R_y)
