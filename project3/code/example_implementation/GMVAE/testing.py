import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch'))

import torch
from torch.utils.data import TensorDataset, DataLoader
from model.GMVAE import GMVAE

# Generate patterns - MODIFIED to be more distinct for GMVAE
def circle(n=50):
    t = np.linspace(0, 2*np.pi, n)
    # Circles centered at (0, 0) with consistent radius
    radius = 10.0
    x = radius*np.cos(t) + np.random.randn(n)*0.2
    y = radius*np.sin(t) + np.random.randn(n)*0.2
    return np.column_stack([x, y]).flatten()

def line(n=50):
    # Lines always horizontal, centered at (0, 10), varying length
    t = np.linspace(0, 1, n)
    length = 20.0
    x = -length/2 + length*t
    y = np.ones(n) * 10.0 + np.random.randn(n)*0.2
    return np.column_stack([x, y]).flatten()

def triangle(n=50):
    n_side = n // 3
    # Triangle centered at (0, 20), fixed size
    size = 10.0
    x = np.concatenate([np.linspace(-size, size, n_side), 
                        np.linspace(size, 0, n_side),
                        np.linspace(0, -size, n - 2*n_side)])
    y = np.concatenate([np.ones(n_side)*20,
                        np.linspace(20, 20+size, n_side),
                        np.linspace(20+size, 20, n - 2*n_side)])
    x += np.random.randn(n)*0.2
    y += np.random.randn(n)*0.2
    return np.column_stack([x, y]).flatten()

def zigzag(n=50):
    x = np.linspace(0, 20, n)
    y = 8 * np.sin(8 * 2 * np.pi * np.linspace(0, 1, n))
    return np.column_stack([x, y]).flatten()

def spiral(n=50):
    t = np.linspace(0, 6*np.pi, n)
    r = t / (6*np.pi) * 15
    return np.column_stack([r*np.cos(t), r*np.sin(t)]).flatten()

# Create training data
print("Generating data...")
train_data = []
train_labels = []
training_samples = 500
for i in range(training_samples):
    train_data.extend([circle(), line(), triangle()])
    train_labels.extend([0, 1, 2])

train_data = np.array(train_data, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int64)

# Create test data BEFORE normalization
zigzag_data_raw = zigzag()
spiral_data_raw = spiral()
line_data_raw = line()

# NO NORMALIZATION - keep raw data for better pattern separation
# GMVAE has sigmoid output, so we need data in roughly [0, 1] range
# Instead of normalizing all together, shift and scale to preserve differences
all_data = np.vstack([train_data, zigzag_data_raw, spiral_data_raw, line_data_raw])
data_min = all_data.min()
data_max = all_data.max()
data_range = data_max - data_min

# Scale to [0, 1] but preserve the relative differences between patterns
train_data = (train_data - data_min) / data_range
zigzag_data = (zigzag_data_raw - data_min) / data_range
spiral_data = (spiral_data_raw - data_min) / data_range
line_data = (line_data_raw - data_min) / data_range

print(f"Train data shape: {train_data.shape}")
print(f"Data range: [{data_min:.2f}, {data_max:.2f}] -> [0, 1]")
print(f"Circle mean: {train_data[train_labels==0].mean():.4f}, std: {train_data[train_labels==0].std():.4f}")
print(f"Line mean: {train_data[train_labels==1].mean():.4f}, std: {train_data[train_labels==1].std():.4f}")
print(f"Triangle mean: {train_data[train_labels==2].mean():.4f}, std: {train_data[train_labels==2].std():.4f}")

# Model setup - using supervised labels to help the model learn clusters
class Args:
    epochs = 500
    cuda = 0
    verbose = 0  # Less verbose output
    batch_size = 32
    batch_size_val = 32
    learning_rate = 1e-3
    decay_epoch = -1
    lr_decay = 0.5
    w_categ = 50.0  # High categorical weight to encourage cluster separation
    w_gauss = 1.0
    w_rec = 1.0
    rec_type = 'mse'
    num_classes = 3
    gaussian_size = 64
    input_size = train_data.shape[1]
    init_temp = 1.0
    decay_temp = 0  # Keep temperature constant
    hard_gumbel = 0
    min_temp = 0.5
    decay_temp_rate = 0.013862944
    optimizer = 'Adam'
    optimizer_params = {}

args = Args()

# Train
print("Training...")
train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

gmvae = GMVAE(args)
gmvae.train(train_loader, val_loader)

# Test
print("\nTesting...")
gmvae.network.eval()

with torch.no_grad():
    zigzag_sample = torch.FloatTensor(zigzag_data).unsqueeze(0)
    spiral_sample = torch.FloatTensor(spiral_data).unsqueeze(0)
    line_sample = torch.FloatTensor(line_data).unsqueeze(0)
    
    train_recon = gmvae.network(torch.FloatTensor(train_data), gmvae.gumbel_temp, gmvae.hard_gumbel)['x_rec'].cpu().numpy()
    zigzag_recon = gmvae.network(zigzag_sample, gmvae.gumbel_temp, gmvae.hard_gumbel)['x_rec'].cpu().numpy()[0]
    spiral_recon = gmvae.network(spiral_sample, gmvae.gumbel_temp, gmvae.hard_gumbel)['x_rec'].cpu().numpy()[0]
    line_recon = gmvae.network(line_sample, gmvae.gumbel_temp, gmvae.hard_gumbel)['x_rec'].cpu().numpy()[0]

# Create figs directory
os.makedirs('figs_test', exist_ok=True)

# Circle plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for i, traj in enumerate(train_data):
    if train_labels[i] == 0:
        traj_2d = traj.reshape(-1, 2)
        ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'b-', linewidth=0.5, alpha=0.2)
for i, traj in enumerate(train_recon):
    if train_labels[i] == 0:
        traj_2d = traj.reshape(-1, 2)
        ax2.plot(traj_2d[:, 0], traj_2d[:, 1], 'b-', linewidth=0.5, alpha=0.2)
ax1.set_title('Circle (Real)')
ax2.set_title('Circle (Reconstructed)')
ax1.axis('equal')
ax2.axis('equal')
plt.tight_layout()
plt.savefig('figs_test/circle.pdf')
plt.close()

# Triangle plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for i, traj in enumerate(train_data):
    if train_labels[i] == 2:
        traj_2d = traj.reshape(-1, 2)
        ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'purple', linewidth=0.5, alpha=0.2)
for i, traj in enumerate(train_recon):
    if train_labels[i] == 2:
        traj_2d = traj.reshape(-1, 2)
        ax2.plot(traj_2d[:, 0], traj_2d[:, 1], 'purple', linewidth=0.5, alpha=0.2)
ax1.set_title('Triangle (Real)')
ax2.set_title('Triangle (Reconstructed)')
ax1.axis('equal')
ax2.axis('equal')
plt.tight_layout()
plt.savefig('figs_test/triangle.pdf')
plt.close()

# Line plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for i, traj in enumerate(train_data):
    if train_labels[i] == 1:
        traj_2d = traj.reshape(-1, 2)
        ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'g-', linewidth=0.5, alpha=0.2)
for i, traj in enumerate(train_recon):
    if train_labels[i] == 1:
        traj_2d = traj.reshape(-1, 2)
        ax2.plot(traj_2d[:, 0], traj_2d[:, 1], 'g-', linewidth=0.5, alpha=0.2)
ax1.set_title('Line (Real)')
ax2.set_title('Line (Reconstructed)')
ax1.axis('equal')
ax2.axis('equal')
plt.tight_layout()
plt.savefig('figs_test/line.pdf')
plt.close()

# Test data plot
fig, ax = plt.subplots(figsize=(10, 8))
zigzag_2d = zigzag_data.reshape(-1, 2)
spiral_2d = spiral_data.reshape(-1, 2)
line_2d = line_data.reshape(-1, 2)
zigzag_recon_2d = zigzag_recon.reshape(-1, 2)
spiral_recon_2d = spiral_recon.reshape(-1, 2)
line_recon_2d = line_recon.reshape(-1, 2)

ax.plot(zigzag_2d[:, 0], zigzag_2d[:, 1], 'r-', linewidth=2, label='Zigzag (real)')
ax.plot(zigzag_recon_2d[:, 0], zigzag_recon_2d[:, 1], 'r--', linewidth=2, alpha=0.6, label='Zigzag (recon)')
ax.plot(spiral_2d[:, 0], spiral_2d[:, 1], 'orange', linewidth=2, label='Spiral (real)')
ax.plot(spiral_recon_2d[:, 0], spiral_recon_2d[:, 1], 'orange', linestyle='--', linewidth=2, alpha=0.6, label='Spiral (recon)')
ax.plot(line_2d[:, 0], line_2d[:, 1], 'cyan', linewidth=2, label='Line (real)')
ax.plot(line_recon_2d[:, 0], line_recon_2d[:, 1], 'cyan', linestyle='--', linewidth=2, alpha=0.6, label='Line (recon)')

ax.set_title('Test Data')
ax.legend()
ax.axis('equal')
plt.tight_layout()
plt.savefig('figs_test/test.pdf')
plt.close()

print("\nSaved plots to figs_test/ directory")
