from pandas import DataFrame


def normalize_dataframe(df: DataFrame) -> DataFrame:
    """
    Perform Min-Max normalization to scale values from 0 to 1.

    Normalization helps improve the performance
    of machine learning algorithms by ensuring that
    features on different scales do not bias the results

    Normalized value = (x - min) / (max - min)
    """
    return (df - df.min()) / (df.max() - df.min())


def calculate_mean(data, n):
    total_sum = 0

    for value in data:
        total_sum += value

    mean = total_sum / n
    return mean


def calculate_slope(df, n, km_mean, price_mean):
    numerator = 0
    denominator = 0

    for index in range(n):
        numerator += (df['km'][index] - km_mean) * \
            (df['price'][index] - price_mean)
        denominator += (df['km'][index] - km_mean) ** 2

    return numerator / denominator


def calculate_intercept(km_mean, price_mean, slope):
    return price_mean - slope * km_mean


def linear_function(x, slope, intercept):
    return slope * x + intercept
