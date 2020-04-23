import matplotlib.pyplot as plt
def plot_column(X, x_pos, y_pos, x_label, y_label):
    plt.scatter(X[:, x_pos],
                X[:, y_pos],
                cmap='viridis')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_map(x, y, x_label, y_label, title):
    plt.scatter(x,
                y,
                cmap='viridis')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    plt.savefig(title)