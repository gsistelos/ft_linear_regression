#!/usr/bin/env python3

import pandas as pd
import math_utils as mu
import plot_utils as pu


if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    normalized_df = mu.z_score_normalization(df, df.mean(), df.std())
    m = len(df)

    normalized_theta_0, normalized_theta_1 = mu.gradient_descent(
        normalized_df['km'], normalized_df['price'], m)

    x = df['km']
    y = df['price']

    theta_1 = normalized_theta_1 * (y.std() / x.std())
    theta_0 = y.mean() - (theta_1 * x.mean())

    pu.plot(x, y, theta_0, theta_1)
