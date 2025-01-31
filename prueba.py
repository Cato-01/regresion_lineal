# 1. Import standard libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# 2. Seed for reproducibility
np.random.seed(4500)

# 3. Generate synthetic data
x = np.arange(0,100, 1)
noise_mean, noise_std = 0, 5
true_coeffs = [1, 3]
y = true_coeffs[1]*x + true_coeffs[0] + np.random.normal(loc=noise_mean, scale=noise_std, size=len(x))

# 4. Visualise the generated synthetic dataset
plt.figure(figsize=(10,7))
plt.scatter(x, y, label='Synthetic dataset')
plt.xlabel(r"$x$", fontsize=20)
plt.ylabel("$f_{\mathbf{w}}(x)$", fontsize=20)
plt.title(rf"$f_{{\mathbf{{w}}}}(x) = {true_coeffs[1]} x + {true_coeffs[0]} + \epsilon$, where $\epsilon \sim \mathcal{{N}}(\mu=0, \sigma={noise_std})$", fontsize=20)
plt.legend()
plt.show()
