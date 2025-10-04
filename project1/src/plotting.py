import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figs", exist_ok=True)


def setup_plot_formatting():
    """Apply standard formatting to current plot."""
    plt.grid(True, alpha=0.6)

    # Only add legend if there are labeled artists
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:  # Only create legend if there are labeled items
        plt.legend(fontsize=16, framealpha=0.6)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()



def mse_degree_ols(results):
    """
    Plot MSE vs polynomial degree for training and test data.

    Parameters
    ----------
    results_dict : dict
        Dictionary where keys are sample sizes and values are results dictionaries
    sample_size : int
        Sample size to plot
    """
    

    plt.figure(figsize=(8, 6))
    plt.plot(
        results["degrees"],
        results["train_mse"],
        "o-",
        label="MSE (train)",
        linewidth=2,
        markersize=6,
        color='darkviolet'
    )
    plt.plot(
        results["degrees"],
        results["test_mse"],
        "o-",
        label="MSE (test)",
        linewidth=2,
        markersize=6,
        color='#D63290'
    )

    min_test_mse = min(results["test_mse"])
    min_degree = results["degrees"][results["test_mse"].index(min_test_mse)]
    plt.plot(min_degree, min_test_mse, '*', markersize=20, markeredgewidth=2, color='#F2B44D', markeredgecolor='black',
         label=f'Min Test MSE: {min_test_mse:.2f}') 
    
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"MSE", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/mse_vs_degree_ols.pdf", bbox_inches='tight')
    plt.show()


def r2_degree_ols(results):
    """
    Plot R² vs polynomial degree for training and test sets.

    Parameters
    ----------
    results : dict
        Results dictionary containing 'degrees', 'train_r2', 'test_r2'
    sample_size : int
        Sample size for labeling
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["degrees"],
        results["train_r2"],
        "o-",
        label="R² (train)",
        linewidth=2,
        markersize=6,
        color='darkviolet'
    )
    plt.plot(
        results["degrees"],
        results["test_r2"],
        "o-",
        label="R² (test)",
        linewidth=2,
        markersize=6,
        color='#D63290'
    )



    max_test_r2 = max(results["test_r2"])
    max_degree = results["degrees"][results["test_r2"].index(max_test_r2)]

    plt.plot(max_degree, max_test_r2, '*', markersize=20, markeredgewidth=2, color='#F2B44D', markeredgecolor='black',
         label=f'Max Test R²: {max_test_r2:.2f}')
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"R²", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/r2_vs_degree_ols.pdf", bbox_inches='tight')
    plt.show()


def plot_theta(theta: list, method: str):
    """
    Plot the model parameters (theta) for a given method.
    
    Parameters
    ----------
    theta : list or np.ndarray
        Model parameters to plot
    method : str
        Method name for labeling the plot
    """

    plt.figure(figsize=(8, 6))
    setup_plot_formatting()
    for i, theta in enumerate(theta):
        for param in theta:
            plt.scatter(i+1, param, label=f"θ_{i}" if i == 0 else "", s=100, color='mediumorchid', alpha=0.5, edgecolors='black')
    plt.xlabel('Polynomial Degree', fontsize=16)
    plt.ylabel(rf'$\theta_{{{method}}}$', fontsize=16)
    plt.savefig(f"figs/theta_{method.lower()}.pdf", bbox_inches='tight')
    plt.show()





def mse_degree_ridge(results):
    """
    Plot MSE vs polynomial degree for a specific lambda.
    
    Parameters
    ----------
    results_for_N : dict
        Dictionary where keys are lambda values
    lam : float
        Lambda value to plot
    """
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["degrees"],
        results["train_mse"],
        "o-",
        label="MSE (train)",
        linewidth=2,
        markersize=6,
        color='darkviolet'
    )
    plt.plot(
        results["degrees"],
        results["test_mse"],
        "o-",
        label="MSE (test)",
        linewidth=2,
        markersize=6,
        color='#D63290'
    )

    lam = results['lambda']
    min_test_mse = min(results["test_mse"])
    min_degree = results["degrees"][results["test_mse"].index(min_test_mse)]
    plt.plot(min_degree, min_test_mse, '*', markersize=20, markeredgewidth=2, color='#F2B44D', markeredgecolor='black',
         label=f'Min Test MSE: {min_test_mse:.2f}') 
    
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    setup_plot_formatting()
    plt.savefig(f"figs/mse_vs_degree_ridge.pdf", bbox_inches='tight')
    plt.show()


def r2_degree_ridge(results):
    """
    Plot R² vs polynomial degree - takes results dict directly.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["degrees"],
        results["train_r2"],
        "o-",
        label="R² (train)",
        linewidth=2,
        markersize=6,
        color='darkviolet'
    )
    plt.plot(
        results["degrees"],
        results["test_r2"],
        "o-",
        label="R² (test)",
        linewidth=2,
        markersize=6,
        color='#D63290'
    )

    max_test_r2 = max(results["test_r2"])
    max_degree = results["degrees"][results["test_r2"].index(max_test_r2)]

    plt.plot(max_degree, max_test_r2, '*', markersize=20, markeredgewidth=2, color='#F2B44D', markeredgecolor='black',
         label=f'Max Test R²: {max_test_r2:.2f}')
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel("R²", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/r2_vs_degree_ridge.pdf", bbox_inches='tight')
    plt.show()























def compare(x, y_noise, y_true, solutions, sample_size, degree, lam, type=None, test=False):
    """
    Plot true function and model predictions (analytical/sklearn + GD).
    
    Parameters
    ----------
    type : str
        Type of regression: 'lasso' for Lasso regression, otherwise treats as OLS/Ridge
    test : bool
        If True, highlights test data points in a different color
    """
    # Use type parameter to determine what baseline to expect
    if type == 'lasso':
        y_scikit, y_gd, x_plotting = solutions
        lasso = True
    else:
        y_analytical, y_gd, x_plotting = solutions
        lasso = False

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")
    
    if test:
        test_mask = np.isin(x, x_plotting)
        train_mask = ~test_mask
        
        plt.scatter(x[train_mask], y_noise[train_mask], color='lightgray', alpha=0.5, s=50, label=r"$y_{noise}$ (train)")
        plt.scatter(x[test_mask], y_noise[test_mask], color='dimgray', alpha=0.7, s=50, label=r"$y_{noise}$ (test)")
    else:
        plt.scatter(x, y_noise, color='lightgray', alpha=0.7, s=50, label=r"$y_{noise}$")
    
    if lasso:
        plt.plot(x_plotting, y_scikit, label=f"Scikit-Learn, λ={lam:.0e}")
    else:
        plt.plot(x_plotting, y_analytical, label=f"Analytical, λ={lam:.0e}")
    
    plt.plot(x_plotting, y_gd, label=f"GD, λ={lam:.0e}")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=16)
    
    title = f'Test split - {type if type else "Comparison"}' if test else f'Full dataset - {type if type else "Comparison"}'
    plt.title(title, fontsize=16)
    plt.show()


def compare_gd(x, y_noise, y_true, solutions, sample_size, degree, lam, type=None, test=False):
    """
    Plotting solutions using different methods for computing the optimal parameters (gradient descent).
    
    Parameters
    ----------
    test : bool
        If True, highlights test data points in a different color
    """
    
    if type == 'lasso':
        y_scikit, y_gd, y_momentum, y_adagrad, y_rmsprop, y_adam, x_plotting = solutions
        lasso = True
    else:
        y_analytical, y_gd, y_momentum, y_adagrad, y_rmsprop, y_adam, x_plotting = solutions
        lasso = False

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")
    
    if test:
        # Find which points from the full dataset are NOT in the test set (i.e., training points)
        # Create a mask for test points
        test_mask = np.isin(x, x_plotting)
        train_mask = ~test_mask
        
        # Plot training points in light gray
        plt.scatter(x[train_mask], y_noise[train_mask], color='lightgray', alpha=0.5, s=50, label=r"$y_{noise}$ (train)")
        # Plot test points in a different color to highlight them
        plt.scatter(x[test_mask], y_noise[test_mask], color='dimgray', alpha=0.7, s=50, label=r"$y_{noise}$ (test)")
    else:
        # Plot all data points in one color
        plt.scatter(x, y_noise, color='lightgray', alpha=0.7, s=50, label=r"$y_{noise}$")
    
    if lasso:
        plt.plot(x_plotting, y_scikit, label="Scikit-Learn")
    else:
        plt.plot(x_plotting, y_analytical, label="Analytical")

    plt.plot(x_plotting, y_gd, label="Gradient Descent")
    plt.plot(x_plotting, y_momentum, label="Momentum")
    plt.plot(x_plotting, y_adagrad, label="AdaGrad")
    plt.plot(x_plotting, y_rmsprop, label="RMSProp")
    plt.plot(x_plotting, y_adam, label="Adam")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=14)
    title = f'Test split - GD Methods {type if type else ""}' if test else f'Full dataset - GD Methods {type if type else ""}'
    plt.title(title, fontsize=16)
    plt.show()


def compare_sgd(x, y_noise, y_true, solutions, sample_size, degree, lam, type=None, test=False):
    """
    Plot true function and model predictions (analytical/sklearn + GD + SGD).
    
    Parameters
    ----------
    type : str
        Type of regression: 'lasso' for Lasso regression, otherwise treats as OLS/Ridge
    test : bool
        If True, highlights test data points in a different color
    """
    # Use type parameter to determine what baseline to expect
    if type == 'lasso':
        y_pred_sklearn, y_pred_gd, y_pred_sgd, x_plotting = solutions
        lasso = True
    else:
        y_pred_analytical, y_pred_gd, y_pred_sgd, x_plotting = solutions
        lasso = False

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")
    
    if test:
        test_mask = np.isin(x, x_plotting)
        train_mask = ~test_mask
        
        plt.scatter(x[train_mask], y_noise[train_mask], color='lightgray', alpha=0.5, s=50, label=r"$y_{noise}$ (train)")
        plt.scatter(x[test_mask], y_noise[test_mask], color='dimgray', alpha=0.7, s=50, label=r"$y_{noise}$ (test)")
    else:
        plt.scatter(x, y_noise, color='lightgray', alpha=0.7, s=50, label=r"$y_{noise}$")
    
    if lasso:
        plt.plot(x_plotting, y_pred_sklearn, label=f"Scikit-Learn, λ={lam:.0e}")
    else:
        plt.plot(x_plotting, y_pred_analytical, label=f"Analytical, λ={lam:.0e}")
    
    plt.plot(x_plotting, y_pred_gd, label=f"GD, λ={lam:.0e}")
    plt.plot(x_plotting, y_pred_sgd, label=f"SGD, λ={lam:.0e}")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=14)
    
    title = f'Test split - SGD Methods {type if type else ""}' if test else f'Full dataset - SGD Methods {type if type else ""}'
    plt.title(title, fontsize=16)
    plt.show()


