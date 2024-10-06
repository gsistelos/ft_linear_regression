#!/usr/bin/env python3

import pandas as pd
import math_utils as mu
import plot_utils as pu


if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    df = mu.z_score_normalization(df, df.mean(), df.std())
    n = len(df)

    x = df['km']
    y = df['price']

    x_mean = mu.calculate_mean(x, n)
    y_mean = mu.calculate_mean(y, n)

    intercept, slope = mu.gradient_descent(x, y)

    pu.plot(x, y, slope, intercept)
