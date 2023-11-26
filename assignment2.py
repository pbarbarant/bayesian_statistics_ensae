# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde

plt.style.use("fivethirtyeight")
PARALLEL = True

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


def sample_posterior_marginal_q(X, eps, s, R_y, k=100, T=200):
    return np.random.randn(100_000)


def plot_posterior_median_q(medians, s, R_y):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    # Add histogram
    ax.hist(medians, bins=50, density=True)
    # Add kernel density estimation
    kde = gaussian_kde(medians)
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
    kde = gaussian_kde(q)
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
    kde1 = gaussian_kde(medians)
    x = np.linspace(medians.min(), medians.max(), 1000)
    ax1.plot(x, kde1(x), label="KDE")
    kde2 = gaussian_kde(q)
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
                delayed(sample_posterior_marginal_q)(
                    X[i, :], eps[i, :], s, R_y
                ) for i in range(X.shape[0])
            )
            q = np.stack(q, axis=0)
        else:
            for i in range(X.shape[0]):
                q[i, :] = sample_posterior_marginal_q(X[i, :], eps[i, :], s, R_y)
        
        posterior_median_q = np.median(q, axis=1)

        make_plots(q[-1, :], posterior_median_q, s, R_y)
