import numpy as np

# 读取保存的 .npy 文件
predictions = np.load("lstm_final_predictions_output96.npy")
ground_truth = np.load("lstm_final_ground_truths_output96.npy")

# 检查文件内容和形状
print(f"Predictions shape: {predictions.shape}")
print(f"Ground Truths shape: {ground_truth.shape}")

# 显示部分数据（前5条记录）
print("\nPredictions (first 2 samples):")
print(predictions[:2])

print("\nGround Truths (first 2 samples):")
print(ground_truth[:2])
