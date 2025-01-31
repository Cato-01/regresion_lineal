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

# 5. Split the data into training and testing sets using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# 6. Visualise the generated train and test synthetic dat sets
plt.figure(figsize=(10,7))
plt.scatter(x_test, y_test, label='Testing dataset')
plt.scatter(x_train, y_train, label='Training dataset', color=[1,0,0])
plt.xlabel(r"$x$", fontsize=20)
plt.ylabel("$f_{\mathbf{w}}(x)$", fontsize=20)
plt.title(rf"$f_{{\mathbf{{w}}}}(x) = {true_coeffs[1]} x + {true_coeffs[0]} + \epsilon$, where $\epsilon \sim \mathcal{{N}}(\mu=0, \sigma={noise_std})$", fontsize=20)
plt.legend()
plt.show()

# 7. Build the linear regression model using a multiple-input single neuron
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)])

# 8. Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=200)
model.compile(optimizer=opt, loss='mse')

# 9. Add EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)

# 10. Train the model
start_time = time.time()
model.fit(x_train, y_train, epochs=3000, batch_size=5, validation_split=0.10, verbose=1, callbacks=[early_stopping])
print(f"\nElapsed time: {time.time() - start_time} sec")

# 11. Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f'\nTest Loss: {loss}')

# 12. Print the weights of the trained model
weights = model.layers[0].get_weights()
print(f"Weights [w1]: {weights[0]}")
print(f"Bias [w0]: {weights[1]}")

# 13. Make predictions
y_pred_test = model.predict(x_test)

# 14. Plot the results
plt.figure(figsize=(10, 7))
plt.scatter(x_train, y_train, label='Training data')
plt.plot(x_test, y_pred_test, label='Model', linewidth=3, color=[1,0,0])
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel("$\hat{f}_{\mathbf{w}}(x)$", fontsize=20)
plt.legend()
plt.show()

model.summary()