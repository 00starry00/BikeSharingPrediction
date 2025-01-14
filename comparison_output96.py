import matplotlib.pyplot as plt
import numpy as np

def plot_mse_mae_comparison():
    """
    Plot MSE and MAE comparisons for different models.
    """
    # 模型名称
    models = ['LSTM', 'Transformer', 'TFT', 'Autoformer']

    # MSE 和 MAE 数据
    mse_means = [0.0264, 0.0254, 0.0279, 0.0201]
    mse_stds = [0.0042, 0.0056, 0.0012, 0.0013]

    mae_means = [0.1048, 0.1067, 0.1165, 0.0996]
    mae_stds = [0.0091, 0.0133, 0.0027, 0.0037]

    # 绘制 MSE 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(models, mse_means, yerr=mse_stds, capsize=5, color=['blue', 'orange', 'green', 'red'])
    plt.title('MSE Comparison (Output=96)')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Models')
    plt.grid(axis='y')
    plt.savefig('mse_comparison_output96.png')  # 保存 MSE 图
    plt.show()

    # 绘制 MAE 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(models, mae_means, yerr=mae_stds, capsize=5, color=['blue', 'orange', 'green', 'red'])
    plt.title('MAE Comparison (Output=96)')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Models')
    plt.grid(axis='y')
    plt.savefig('mae_comparison_output96.png')  # 保存 MAE 图
    plt.show()

if __name__ == "__main__":
    plot_mse_mae_comparison()
