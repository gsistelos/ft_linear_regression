from typing import List
import matplotlib.pyplot as plt


def plot_linear_function(
    x: List[float], y: List[float],
    slope: float, intercept: float
) -> None:
    """
    Plot the linear function y = mx + b
    """
    plt.scatter(x, y)

    plt.plot(x, slope * x + intercept, color='red')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')

    plt.title('Linear Regression with Gradient Descent')
    plt.show()
