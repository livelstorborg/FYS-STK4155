# Project 2: Comparison of Feed Forward Neural Networks with Classical Polynomial Regression Methods

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug

## Description
In this project we comapre feed forward neural networks with different hyperparameters, architectures, optimizers and regularization with classical linear regression methods. We analyze the best combinations of the neural network architecture and measure them against how the classical polynomial linear regression approaches perform. We also analyze the performance of the neural networks on the MNIST dataset, to see how well it performs in picture classification.

## Project Structure

```
├── project2
│   ├── code
│   │   ├── notebooks
│   │   │   ├── figs
│   │   │   ├── Makefile
│   │   │   └── results.ipynb
│   │   └── src
│   │       ├── __init__.py
│   │       ├── activations.py
│   │       ├── losses.py
│   │       ├── metrics.py
│   │       ├── neural_network.py
│   │       ├── optimizers.py
│   │       ├── plotting.py
│   │       ├── training.py
│   │       ├── utils_b.py
│   │       ├── utils_d.py
│   │       ├── utils_e.py
│   │       ├── utils_f.py
│   │       └── utils.py
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
└── README.md
├── report.pdf

```

## Setup and Installation

### Prerequisites
- Python 3.11 (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project2
   ```

2. **Install uv** (if not already installed)
   ```bash
   # macOS/Linux
   brew install uv
   
   # Or using curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or with pip
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```
   This creates a virtual environment in `.venv/` and installs all required packages.

## Usage

### Running Individual Analysis Scripts

```bash
# Part x: 
uv run code/main.py
```



### Running the Jupyter Notebook

To reproduce all results and generate all figures in one place:
```bash
uv run jupyter notebook code/results.ipynb
```
additionally, you can run each cell in the notebook individually to generate figures for each problem.

## Key Features

- **Part a**: Analytical warm-up  
  - Define the cost/loss functions (MSE, binary cross-entropy, multiclass cross-entropy) and derive their gradients. :contentReference[oaicite:0]{index=0}  
  - Define the activation functions (Sigmoid, ReLU, Leaky ReLU) and their derivatives. :contentReference[oaicite:1]{index=1}

- **Part b**: Writing your own Neural Network code  
  - Implement a feed-forward neural network (FFNN) for regression using the 1D Runge function \(f(x) = 1/(1+25x^2)\). :contentReference[oaicite:2]{index=2}  
  - Use MSE as cost, Sigmoid activation in hidden layers, flexible number of hidden layers/nodes, random weight initialization (normal distribution). :contentReference[oaicite:3]{index=3}  
  - Compare your results with OLS regression from Project 1 and explore architecture/learning-rate effects. :contentReference[oaicite:4]{index=4}

- **Part c**: Testing against other software libraries  
  - Benchmark your FFNN implementation against existing libraries (e.g., Scikit‑Learn, TensorFlow/Keras, PyTorch). :contentReference[oaicite:9]{index=9}  
  - Optionally test gradient correctness via automatic differentiation (e.g., via JAX or Autograd). :contentReference[oaicite:12]{index=12}

- **Part d**: Testing different activation functions and depths of the neural network  
  - Experiment with hidden-layer activation functions (Sigmoid, ReLU, Leaky ReLU). :contentReference[oaicite:13]{index=13}  
  - Vary the number of hidden layers and nodes, and assess signs of overfitting. :contentReference[oaicite:14]{index=14}

- **Part e**: Testing different norms  
  - Introduce regularization (L1 and L2 norms) in the cost function for the regression part (1D Runge function). :contentReference[oaicite:15]{index=15}  
  - Compare results with Ridge (L2) and Lasso (L1) regression from Project 1. :contentReference[oaicite:16]{index=16}

- **Part f**: Classification analysis using neural networks  
  - Adapt your FFNN to perform multi-class classification, e.g., on the full MNIST dataset (or another dataset). :contentReference[oaicite:18]{index=18}  
  - Use Softmax cross-entropy cost for classification. :contentReference[oaicite:19]{index=19}  
  - Explore hyperparameters (learning rate, regularization, network architecture, activation functions) and compare with logistic regression if time permits. :contentReference[oaicite:20]{index=20}

- **Part g**: Critical evaluation of the various algorithms  
  - Summarize and critically evaluate the methods you implemented: which algorithm works best for regression vs classification? What are the advantages/disadvantages? :contentReference[oaicite:21]{index=21}  
  - Discuss the impact of architecture choices, activation functions, regularization, gradient methods, etc. :contentReference[oaicite:22]{index=22}


## Output

All generated figures are saved in a folder `figs/`, that is generated automatically when running the cells in the notebook.


## Dependencies

Main packages (managed via `pyproject.toml`):
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning utilities and comparison
- `pandas` - Data manipulation
- `seaborn` - Statistical data visualization

Development packages:
- `jupyter` / `notebook` - Interactive notebooks
- `ipython` - Enhanced Python shell
- `ipykernel` - Jupyter kernel for Python

### Adding New Dependencies

To add a new package:
```bash
uv add <package-name>       # Adds to dependencies
uv add --dev <package-name> # Adds to dev dependencies
```

## Course Information

**Course**: FYS-STK3155/FYS-STK4155 - Applied Data Analysis and Machine Learning  
**Institution**: University of Oslo  
