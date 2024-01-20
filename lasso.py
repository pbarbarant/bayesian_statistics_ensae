# %%
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from utils import preprocess_dataset, confidence_interval, plot_INDPRO_RATE


if __name__ == "__main__":
    df, timestamp = preprocess_dataset()
    # Regress INDPRO_RATE on all other variables using lasso
    X = df.iloc[1:, :-1]
    y = df.iloc[1:, -1]
    # Standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Run lasso with 10-fold cross validation
    lasso = linear_model.LassoCV(cv=10).fit(X, y)

    # Print R^2
    df_lasso = pd.DataFrame(
        {"Variable": df.columns[:-1], "Coefficient": lasso.coef_}
    )
    # Save non-zero coefficients to csv with no index
    df_lasso[df_lasso["Coefficient"] != 0].to_csv(
        "coefficients/lasso.csv", index=False
    )

    # Do prediction for INDPRO_RATE with confidence interval
    # Get prediction
    pred = lasso.predict(X)
    ci = confidence_interval(y, pred, X)

    plot_INDPRO_RATE(
        actual=df["INDPRO_RATE"][1:],
        pred=pred,
        ci=ci,
        timestamp=timestamp,
        title="Rate of Change in Industrial Production Index (INDPRO)\n Actual vs. Predicted Values using Lasso Regression",
    )
