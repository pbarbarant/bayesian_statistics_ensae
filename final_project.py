# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from assignment2 import one_gibbs_iteration
from utils import (
    credible_regions,
    plot_distribution,
    plot_INDPRO_RATE,
    preprocess_dataset,
)


def init_params(X, y):
    """Initialize parameters for the Gibbs sampler"""
    T, k = X.shape

    # Initialize q using beta distribution
    q = stats.beta(1, 1).rvs()
    s = int(q * k)

    # Initialize R2 using beta distribution
    R2 = stats.beta(1, 1).rvs()

    # Initialize z as a random vector of s ones and k-s zeros
    z = np.array([0] * (k - s) + [1] * s)
    np.random.shuffle(z)

    # Initialize beta as a random vector of k values
    beta = (np.random.randn(k) * z).reshape(-1, 1)
    eps = y - X @ beta
    sigma2 = np.var(eps)

    # Sanity checks
    assert eps.shape == (T, 1)
    assert beta.shape == (k, 1)
    assert z.shape == (k,)
    assert R2 > 0 and R2 < 1
    assert q > 0 and q < 1
    assert sigma2 > 0

    return beta, sigma2, R2, q, z, k, T


def gibbs_sampling(X, y, N_iter):
    """Run the Gibbs sampler for N_iter iterations"""
    # Initialize parameters
    q_chain = np.zeros(N_iter)
    beta_chain = np.zeros((N_iter, X.shape[1]))
    y_pred_chain = np.zeros((N_iter, y.shape[0]))
    beta, sigma2, R2, q, z, k, T = init_params(X, y)

    # Create grid of R2 and q values
    x = np.arange(0.01, 0.99, 0.01)
    Rs, qs = np.meshgrid(x, x)
    # Run Gibbs sampler
    for i in tqdm(range(N_iter)):
        R2, q, z, sigma2, beta = one_gibbs_iteration(
            X, y, R2, q, z, sigma2, beta, Rs, qs, k, T
        )
        q_chain[i] = q
        y_pred_chain[i] = (X @ beta).reshape(-1)
        beta_chain[i] = beta.reshape(-1)
    return q_chain, y_pred_chain, beta_chain


if __name__ == "__main__":
    N_iter, burn_in = 11_000, 1000

    df, timestamp = preprocess_dataset()
    X = df.iloc[1:, :-1].values
    y = df.iloc[1:, -1].values.reshape(-1, 1)
    # Standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Randomly fill X columns with zeros
    y = scaler.fit_transform(y)

    # Run Gibbs sampler
    q_chain, y_pred_chain, beta_chain = gibbs_sampling(X, y, N_iter)
    # Remove burn-in
    q_chain = q_chain[burn_in:]
    y_pred_chain = y_pred_chain[burn_in:]
    beta_chain = beta_chain[burn_in:]

    # q_chain trace plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(q_chain, color="black")
    plt.xlabel("Iteration")
    plt.ylabel("q")
    plt.title("Trace plot of q")
    plt.savefig("figures/q_chain_trace.png", bbox_inches="tight", dpi=300)

    df_gibbs = pd.DataFrame(
        {
            "Variable": df.columns[:-1],
            "Mean Posterior Coefficient": np.mean(beta_chain, axis=0),
            "Standard Error": np.std(beta_chain, axis=0),
            "5% lower bound": np.quantile(beta_chain, 0.025, axis=0),
            "95% upper bound": np.quantile(beta_chain, 0.975, axis=0),
            "Posterior probability of inclusion": np.mean(
                beta_chain != 0, axis=0
            ),
        },
        index=df.columns[:-1],
    )
    # Sort coefficients by posterior probability of inclusion
    df_gibbs = df_gibbs.sort_values(
        "Posterior probability of inclusion", ascending=False
    )
    # Remove coefficients with posterior probability of inclusion < median of q_chain
    df_gibbs = df_gibbs[df_gibbs["Posterior probability of inclusion"] > 0.5]
    # Set df_gibbs precision to 4 decimal places
    df_gibbs = df_gibbs.round(4)
    # Save non-zero coefficients to csv
    df_gibbs.to_csv("coefficients/gibbs.csv", index=False)

    # Get credible regions
    cr = credible_regions(y_pred_chain)

    # Rescale y, y_pred_chain, and cr
    cr = scaler.inverse_transform(cr)
    y = scaler.inverse_transform(y)
    y_pred_chain = scaler.inverse_transform(y_pred_chain)

    # Calculate R2
    y_pred = y_pred_chain.mean(axis=0)
    correlation_matrix = np.corrcoef(y.reshape(-1), y_pred)
    R2 = correlation_matrix[0, 1] ** 2

    # Plot actual vs. predicted values
    plot_INDPRO_RATE(
        y,
        y_pred,
        cr,
        timestamp,
        regions="cr",
        title=f"Rate of Change in Industrial Production Index (INDPRO)\n Actual vs. Predicted Values using Bayesian Regression, R2 = {R2:.4f}",
    )

    # Plot posterior distribution of q
    plot_distribution(q_chain)
