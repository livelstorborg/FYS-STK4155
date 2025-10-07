import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from src.plotting import *
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge

np.random.seed(42)

def run_experiment(models=['ols', 'ridge', 'lasso'], n_stochastic_runs=10):

    N = 1_000_000
    degree = 5
    lam = 1e-2
    n_epochs = 40
    num_iters = 500

    # Generate data
    x = np.linspace(-1, 1, N)
    random_noise = np.random.normal(0, 0.1, N)
    y_true = runge(x)
    y_noise = y_true + random_noise
    X = polynomial_features(x, degree)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_noise, test_size=0.25, random_state=42
    )
    X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
    X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)
    data = [X_train_s, X_test_s, y_train_s, y_test_s, X_train[:,0], X_test[:,0], y_mean]

    # =============================================================================
    #                    LEARNING RATES
    # =============================================================================
    learning_rates = {
        'gd': 0.36,
        'momentum': 0.1,
        'adagrad': 0.25,
        'rmsprop': 0.01,
        'adam': 0.05,

        'sgd': 0.005,
        'sgd_momentum': 0.001,
        'sgd_adagrad': 0.01,
        'sgd_rmsprop': 0.0005,
        'sgd_adam': 0.0001
    }
    batch_sizes = [256]

    results = []

    # =============================================================================
    #                    RUN EXPERIMENTS
    # =============================================================================
    for model in models:
        # ------ FULL-BATCH METHODS ------
        for opt in ['gd', 'momentum', 'adagrad', 'rmsprop', 'adam']:
            analysis = RegressionAnalysis(
                data,
                degree=degree,
                lam=lam,
                eta=learning_rates[opt],
                num_iters=num_iters,
                full_dataset=False,
                tol_relative=1e-5
            )
            start_time = time.time()
            analysis.fit(models=(model,), opts=opt)
            elapsed_time = time.time() - start_time

            run_key = (model, opt)
            history = analysis.runs[run_key]['history']
            iterations = len(history)
            final_mse = history[-1]
            initial_mse = history[0]

            if final_mse > initial_mse:
                converged = 'Diverged'
                final_mse_display = '-'
            elif iterations == num_iters:
                converged = 'No'
                final_mse_display = f'{final_mse:.6f}'
            else:
                converged = 'Yes'
                final_mse_display = f'{final_mse:.6f}'

            results.append({
                'Model': model.upper(),
                'Method': opt.upper(),
                'Batch_Type': 'Full-Batch',
                'Batch_Size': len(X_train_s),
                'Epochs/Iters': iterations,
                'Time_Seconds': elapsed_time,
                'Final_MSE': final_mse_display,
                'Converged': converged
            })

        # ------ MINI-BATCH SGD METHODS ------
        for opt in ['sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam']:
            base_opt = opt.replace('sgd_', '').replace('sgd', 'gd')
            for batch_size in batch_sizes:
                mse_runs = []
                time_runs = []
                epochs_runs = []

                for _ in range(n_stochastic_runs):
                    analysis = RegressionAnalysis(
                        data,
                        degree=degree,
                        lam=lam,
                        eta=learning_rates[opt],
                        num_iters=None,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        full_dataset=False,
                        tol_relative=1e-5
                    )
                    start_time = time.time()
                    analysis.fit(models=(model,), opts=opt)
                    elapsed_time = time.time() - start_time

                    run_key = (model, opt)
                    history = analysis.runs[run_key]['history']
                    epochs = len(history)
                    final_mse = history[-1]

                    mse_runs.append(final_mse)
                    time_runs.append(elapsed_time)
                    epochs_runs.append(epochs)

                avg_mse = np.mean(mse_runs)
                avg_time = np.mean(time_runs)
                avg_epochs = int(np.ceil(np.mean(epochs_runs)))

                results.append({
                    'Model': model.upper(),
                    'Method': opt.upper(),
                    'Batch_Type': 'Mini-Batch',
                    'Batch_Size': batch_size,
                    'Epochs/Iters': avg_epochs,
                    'Time_Seconds': avg_time,
                    'Final_MSE': f'{avg_mse:.6f}',
                    'Converged': 'Yes' if avg_epochs < n_epochs else 'No'
                })

    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    models_to_run = ['ols']  
    n_stochastic_runs = 30
    df = run_experiment(models=models_to_run, n_stochastic_runs=n_stochastic_runs)

    for model in df['Model'].unique():
        print(f"\n{model}:")
        print("-"*60)
        df_model = df[df['Model'] == model].copy()
        display_cols = ['Method', 'Batch_Type', 'Batch_Size', 'Time_Seconds', 'Epochs/Iters', 'Converged', 'Final_MSE']
        df_model['Time_Seconds'] = df_model['Time_Seconds'].apply(lambda x: f"{x:.2f}")
        print(df_model[display_cols].to_string(index=False))
