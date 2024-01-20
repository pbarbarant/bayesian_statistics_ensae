# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def preprocess_dataset():
    # Open Macro1.csv
    df = pd.read_csv("Macro1.csv")
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how="any")
    # Remove first row
    df = df.iloc[1:]
    # Remove first column
    timestamp = df["sasdate"][::12]
    df = df.iloc[:, 1:]

    # Add INDPRO growth rate column
    df["INDPRO_RATE"] = df["INDPRO"].pct_change()
    return df, timestamp


def confidence_interval(y, pred, X):
    # Get confidence interval
    alpha = 0.05
    n, p = X.shape
    t = 1 - alpha / 2
    # Get standard error
    se = np.sqrt(np.sum((y - pred) ** 2) / (n - p - 1))
    # Get confidence interval
    ci = np.array([pred - t * se, pred + t * se])
    return ci


def credible_regions(y, alpha=0.05):
    """Get credible regions for y"""
    # Get quantiles
    y_lower = np.quantile(y, alpha / 2, axis=0)
    y_upper = np.quantile(y, 1 - alpha / 2, axis=0)
    return y_lower, y_upper


def plot_INDPRO_RATE(actual, pred, ci, timestamp, regions="ci", title=""):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", c="gray", alpha=0.6)
    plt.plot(pred, label="Prediction")
    if regions == "ci":
        plt.fill_between(
            np.arange(0, len(actual)),
            ci[0],
            ci[1],
            alpha=0.4,
            label="95% Confidence\nInterval",
        )
    elif regions == "cr":
        plt.fill_between(
            np.arange(0, len(actual)),
            ci[0],
            ci[1],
            alpha=0.4,
            label="95% Credible\nInterval",
        )
    else:
        raise ValueError("regions must be either 'ci' or 'cr'")
    plt.xticks(
        np.arange(0, len(actual), 12),
        timestamp,
        rotation=45,
    )
    plt.locator_params(axis="x", nbins=10)
    plt.ylabel("INDPRO_RATE")
    # Add 1973 oil crisis
    plt.axvline(x=12 * 14, c="k", linestyle="--")
    plt.text(12 * 14 + 5, 0.03, "1973 Oil Crisis", rotation=90)
    # Add 2007 financial crisis
    plt.axvline(x=12 * 48, c="k", linestyle="--")
    plt.text(12 * 48 + 5, 0.02, "2007 Financial Crisis", rotation=90)
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    plt.title(title)
    fig.savefig(
        f"figures/plot_{regions}.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_distribution(q):
    """
    Plot posterior median of q and marginal
    posterior distribution of q for a given dataset
    """
    fig = plt.figure(figsize=(12, 8))
    # Add histograms
    plt.hist(q, bins=40, density=True, alpha=0.5)
    # Add kernel density estimations for q marginal posterior distribution
    kde = stats.gaussian_kde(q)
    x = np.linspace(q.min(), q.max(), 1000)
    plt.plot(x, kde(x), label="KDE", color="red")
    plt.xlim(0, 1)

    # Add labels and titles
    plt.xlabel("q")
    plt.ylabel("Density")
    plt.title("Histogram of marginal posterior distribution of q")
    plt.legend()
    fig.savefig(
        "figures/macro1_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
