from matplotlib.pyplot import scatter, plot, xlabel, ylabel, title, legend, show


def plot_linear_function(df, x_values, y_values):
    scatter(df['km'], df['price'])

    xlabel('Kilometers')
    ylabel('Price')

    show()

    scatter(df['km'], df['price'],
            label='Normalized Data', color='blue')
    plot(x_values, y_values, label='Fitted Line', color='red')

    xlabel('Normalized km')
    ylabel('Normalized Price')

    title('Linear Fit of Normalized Data')
    legend()

    show()
