import matplotlib.pyplot as plt


def get_model():
    with open('model.txt', 'r') as f:
        theta_0 = float(f.readline())
        theta_1 = float(f.readline())

    return theta_0, theta_1


def plot_data(x, y, title):
    plt.scatter(x, y)

    plt.xlabel('Kilometers')
    plt.ylabel('Price')

    plt.title(title)
    plt.show()


def plot_function(x, y, intercept, slope, title):
    plt.scatter(x, y)

    plt.plot(x, slope * x + intercept, color='red')

    plt.xlabel('Kilometers')
    plt.ylabel('Price')

    plt.title(title)
    plt.show()
