import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_predictions_and_ground_truth(model_name, output_window):
    """
    加载指定模型的预测值和真实值文件。
    """
    predictions_file = f"{model_name.lower()}_final_predictions_output{output_window}.npy"
    ground_truth_file = f"{model_name.lower()}_final_ground_truths_output{output_window}.npy"
    predictions = np.load(predictions_file)
    ground_truth = np.load(ground_truth_file)
    return predictions, ground_truth

def evaluate_model(predictions, ground_truth):
    """
    计算模型的误差指标：MSE 和 MAE。
    """
    mse = mean_squared_error(ground_truth.flatten(), predictions.flatten())
    mae = mean_absolute_error(ground_truth.flatten(), predictions.flatten())
    return mse, mae

def plot_model_comparisons(models, predictions, ground_truths, output_window, sample_index=0):
    """
    绘制不同模型的预测值和真实值的对比图。
    """
    plt.figure(figsize=(12, 6))

    # 绘制真实值
    plt.plot(ground_truths[0][sample_index], label="Ground Truth", color="black", linestyle="--")

    # 绘制每个模型的预测曲线
    for i, model_name in enumerate(models):
        plt.plot(predictions[i][sample_index], label=f"{model_name} Predictions")

    # 图形设置
    plt.title(f"Model Comparison - Output Window {output_window} Hours")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Bike Rentals")
    plt.legend()
    plt.grid()
    plt.savefig(f"model_comparison_output{output_window}.png")
    plt.show()

def main():
    models = ["LSTM", "Transformer", "TFT", "Autoformer"]
    output_windows = [96, 240]  # 短期预测和长期预测
    results = []

    for output_window in output_windows:
        predictions = []
        ground_truths = []

        print(f"\nComparison for Output Window: {output_window} Hours")
        print("=" * 50)

        for model_name in models:
            pred, truth = load_predictions_and_ground_truth(model_name, output_window)
            mse, mae = evaluate_model(pred, truth)
            predictions.append(pred)
            ground_truths.append(truth)
            results.append((model_name, output_window, mse, mae))
            print(f"{model_name}: MSE = {mse:.4f}, MAE = {mae:.4f}")

        # 绘制对比图
        plot_model_comparisons(models, predictions, ground_truths, output_window)

    # 输出最终对比结果表
    print("\nFinal Comparison Results")
    print("=" * 50)
    print("{:<15}{:<15}{:<15}{:<15}".format("Model", "Output Window", "MSE", "MAE"))
    for result in results:
        print("{:<15}{:<15}{:<15.4f}{:<15.4f}".format(*result))

if __name__ == "__main__":
    main()
