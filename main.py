#!/usr/bin/env python3

from pandas import read_csv

from math_utils import normalize_dataframe, calculate_mean, calculate_slope, calculate_intercept, linear_function
from plot_utils import plot_linear_function

if __name__ == "__main__":
    df = read_csv('data.csv')

    df = normalize_dataframe(df)
    n = len(df)

    km_mean = calculate_mean(df['km'], n)
    price_mean = calculate_mean(df['price'], n)

    slope = calculate_slope(df, n, km_mean, price_mean)
    intercept = calculate_intercept(km_mean, price_mean, slope)

    x_values = []
    y_values = []

    for i in range(100):
        x = df['km'].min() + (i / 99) * \
            (df['km'].max() - df['km'].min())
        x_values.append(x)
        y_values.append(linear_function(x, slope, intercept))

    plot_linear_function(df, x_values, y_values)
