import matplotlib.pyplot as plt


def plot_solution(x, u_num, u_true, title=""):
    plt.plot(x, u_num, label="Numerical")
    plt.plot(x, u_true, "--", label="Analytical")
    plt.title(title)
    plt.legend()
    plt.show()
