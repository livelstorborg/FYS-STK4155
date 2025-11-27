from src.fd_scheme import fd_solve
from src.analytical import u_exact
from src.plotting import plot_solution
from src.nn.model import MLP
from src.nn.train import train_pinn
from src.nn.evaluation import compare_nn_and_exact, test_explicit_scheme


if __name__ == "__main__":
    test_explicit_scheme(Nx=10, T=0.5)
    test_explicit_scheme(Nx=100, T=0.5)

    Nx = 100
    T = 0.5
    u_num, x, t = fd_solve(Nx=Nx, T=T, alpha=0.4)
    u_true = u_exact(x, t)
    plot_solution(x, u_num[-1], u_true[-1], title="t=0.5")

    model, losses = train_pinn(layers=[2, 32, 32, 1], steps=10)
    compare_nn_and_exact(model, Nx=Nx, T=T)
