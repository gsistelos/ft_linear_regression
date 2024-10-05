#!/usr/bin/env python3

import pandas as pd
import math_utils as mu


if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    df = mu.normalize_dataframe(df)
    n = len(df)

    x = df['km']
    y = df['price']

    x_mean = mu.calculate_mean(x, n)
    y_mean = mu.calculate_mean(y, n)

    slope = mu.calculate_slope(x, y, x_mean, y_mean, n)
    intercept = mu.calculate_intercept(x_mean, y_mean, slope)

    print(slope, intercept)
