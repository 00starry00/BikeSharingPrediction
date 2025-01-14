import numpy as np
import matplotlib.pyplot as plt

def plot_results(predictions_file, ground_truth_file, title="Prediction vs Ground Truth", output_file="results.png"):
    """
    Plot predictions vs ground truth.
    """
    # Load saved predictions and ground truth
    predictions = np.load(predictions_file)
    ground_truth = np.load(ground_truth_file)

    # Select a random sample or index for detailed comparison
    sample_index = 0  # You can change this to plot a specific sample
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth[sample_index], label="Ground Truth", color="blue")
    plt.plot(predictions[sample_index], label="Predictions", color="orange", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Bike Rentals")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":
    # Short-term prediction
    plot_results(
        "lstm_final_predictions_output96.npy",
        "lstm_final_ground_truths_output96.npy",
        title="LSTM Short-Term Prediction (96 Hours)",
        output_file="short_term_results.png"
    )

    # Long-term prediction
    plot_results(
        "lstm_final_predictions_output240.npy",
        "lstm_final_ground_truths_output240.npy",
        title="LSTM Long-Term Prediction (240 Hours)",
        output_file="long_term_results.png"
    )
