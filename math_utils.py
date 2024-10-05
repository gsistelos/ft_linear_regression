from pandas import DataFrame
from typing import List


def normalize_dataframe(df: DataFrame) -> DataFrame:
    """
    Perform Z-score normalization to standardize the values

    Z-score normalization centers the data around the mean (0)
    and scales it by the standard deviation (1)

    Normalized value = (x - mean) / std
    """
    return (df - df.mean()) / df.std()


def calculate_mean(x: List[float], n: int) -> float:
    """
    Calculate the mean of a list of numbers

    Mean = sum(x) / n
    """
    total_sum = 0

    for value in x:
        total_sum += value

    mean = total_sum / n
    return mean


def calculate_slope(
    x: List[float], y: List[float],
    x_mean: float, y_mean: float,
    n: int,
) -> float:
    """
    Calculate the slope of the regression line

    The slope quantifies how much y changes for a unit change in x

    Slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    """
    numerator = 0
    denominator = 0

    for i in range(n):
        x_deviation = (x[i] - x_mean)
        y_deviation = (y[i] - y_mean)

        numerator += x_deviation * y_deviation
        denominator += x_deviation ** 2

    return numerator / denominator


def calculate_intercept(
    x_mean: float, y_mean: float, slope: float,
) -> float:
    """
    Calculate the intercept of the regression line

    Intercept is the value of y when x is 0

    Intercept = y_mean - slope * x_mean
    """
    return y_mean - slope * x_mean
