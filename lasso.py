# %%
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from utils import confidence_interval, plot_INDPRO_RATE, preprocess_dataset

if __name__ == "__main__":
    df, timestamp = preprocess_dataset()
    X = df.iloc[1:, :-1]
    y = df.iloc[1:, -1]
    # Standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Run lasso with 10-fold cross validation
    lasso = linear_model.LassoCV(cv=10).fit(X, y)

    # Get predictions and confidence intervals
    pred = lasso.predict(X)
    ci = confidence_interval(y, pred, X)
    df_lasso = pd.DataFrame(
        {
            "Variable": df.columns[:-1],
            "Coefficient": lasso.coef_,
            "Standard Error": stats.sem(lasso.coef_),
            "p-value": stats.t.sf(
                np.abs(lasso.coef_) / stats.sem(lasso.coef_), len(X) - 1
            ),
        },
        index=df.columns[:-1],
    )
    # Calculate R2
    R2 = lasso.score(X, y)

    # Save non-zero coefficients to csv with no index
    df_lasso[df_lasso["Coefficient"] != 0].to_csv(
        "coefficients/lasso.csv", index=False
    )
    plot_INDPRO_RATE(
        actual=df["INDPRO_RATE"][1:],
        pred=pred,
        ci=ci,
        timestamp=timestamp,
        title=f"Rate of Change in Industrial Production Index (INDPRO)\n Actual vs. Predicted Values using Lasso Regression, R2 = {R2:.4f}",
    )
