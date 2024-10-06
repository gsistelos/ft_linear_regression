from typing import List


def z_score_normalization(
    x: List[float],
    mean: float,
    std: float
) -> List[float]:
    """
    Perform Z-score normalization to standardize the values

    Z-score normalization centers the data around the mean (0)
    and scales it by the standard deviation (1)
    """
    return (x - mean) / std


def calculate_mean(x: List[float], n: int) -> float:
    return sum(x) / n


def gradient_descent(
    x: List[float],
    y: List[float],
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
) -> tuple[float, float]:
    """
    Perform gradient descent to find the best-fit line for the given data

    The best-fit line is represented by the equation:
    y(x) = theta_0 + (theta_1 * x)

    - theta_0 (intercept): The value of y when x = 0
    - theta_1 (slope): The rate of change in y with respect to x

    Example:
    - Let's say we have the following data:
    - x = [0, 1, 2] (kilometers)
    - y = [1000, 800, 600] (price)
    - theta_0 = 1000 (base price)
    - theta_1 = -200 (increase per kilometer)
    The best-fit line would be: y(x) = 1000 + (-200 * x)

    The estimated price for 3 and 4 kilometers would be:
    - y(3) = 1000 + (-200 * 3) = 400
    - y(4) = 1000 + (-200 * 4) = 200
    """
    theta_0 = 0
    theta_1 = 0

    m = len(y)  # Number of data points

    for _ in range(max_iterations):
        y_pred = theta_0 + theta_1 * x

        error = y_pred - y

        theta_0 -= learning_rate * calculate_mean(error, m)
        theta_1 -= learning_rate * calculate_mean(error * x, m)

    return theta_0, theta_1
