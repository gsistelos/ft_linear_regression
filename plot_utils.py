from typing import List
import matplotlib.pyplot as plt


def plot(
    x: List[float],
    y: List[float],
    slope: float,
    intercept: float
) -> None:
    plt.scatter(x, y)

    plt.plot(x, slope * x + intercept, color='red')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')

    plt.title('Linear Regression with Gradient Descent')
    plt.show()
