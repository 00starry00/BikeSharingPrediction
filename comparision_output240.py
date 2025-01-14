import matplotlib.pyplot as plt
import numpy as np

# Data for plotting (MSE and MAE values for 4 models)
model_names = ['LSTM', 'Transformer', 'TFT', 'Autoformer']
mse_means = [0.0361, 0.0232, 0.0301, 0.0245]
mse_stds = [0.0048, 0.0016, 0.0033, 0.0016]
mae_means = [0.1272, 0.0997, 0.1205, 0.1076]
mae_stds = [0.0099, 0.0042, 0.0082, 0.0049]

# Plot MSE Comparison
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_means, yerr=mse_stds, capsize=5, color=['blue', 'orange', 'green', 'red'])
plt.title('MSE Comparison (Output=240)')
plt.ylabel('Mean Squared Error')
plt.xlabel('Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('mse_comparison_output240.png')
plt.show()

# Plot MAE Comparison
plt.figure(figsize=(10, 6))
plt.bar(model_names, mae_means, yerr=mae_stds, capsize=5, color=['blue', 'orange', 'green', 'red'])
plt.title('MAE Comparison (Output=240)')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('mae_comparison_output240.png')
plt.show()