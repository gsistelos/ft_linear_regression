import matplotlib.pyplot as plt


def plot(x, y, theta_0: float, theta_1: float):
    plt.scatter(x, y)

    plt.plot(x, theta_1 * x + theta_0, color='red')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')

    plt.title('Linear Regression with Gradient Descent')
    plt.show()
