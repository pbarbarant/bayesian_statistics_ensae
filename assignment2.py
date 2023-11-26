# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed


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


def sample_q(X, eps, s, R_y):
    return np.random.randn(100_000)


def plot_posterior_median_q(medians, s, R_y):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(medians, bins=50, density=True)
    ax.set_xlabel("Posterior median of q")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Histogram of posterior median of q with s={s} and R_y={int(R_y*100)}%"
    )
    return fig, ax


def plot_marginal_posterior_q(q, s, R_y):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(q, bins=50, density=True)
    ax.set_xlabel("q")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Histogram of marginal posterior distribution of q with s={s} and R_y={int(R_y*100)}% (last dataset)"
    )
    return fig, ax


if __name__ == "__main__":
    # Question 1
    X, eps = generate_datasets(n_datasets=100)

    # Question 2
    for s, R_y in product([5, 10, 100], [0.02, 0.25, 0.5]):
        # a.
        posterior_median_q = np.zeros(X.shape[0])
        q = np.zeros((X.shape[0], 100_000))

        for i in range(X.shape[0]):
            q[i, :] = sample_q(X[i, :], eps[i, :], s, R_y)
            posterior_median_q[i] = np.median(q[i, :])

        fig, ax = plot_posterior_median_q(posterior_median_q, s, R_y)
        plt.show()

        # b.
        # We select the last dataset
        fig, ax = plot_marginal_posterior_q(q[-1, :], s, R_y)
        plt.show()
