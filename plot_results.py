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
    # LSTM Short-term prediction
    plot_results(
        "lstm_final_predictions_output96.npy",
        "lstm_final_ground_truths_output96.npy",
        title="LSTM Short-Term Prediction (96 Hours)",
        output_file="lstm_short_term_results.png"
    )

    # LSTM Long-term prediction
    plot_results(
        "lstm_final_predictions_output240.npy",
        "lstm_final_ground_truths_output240.npy",
        title="LSTM Long-Term Prediction (240 Hours)",
        output_file="lstm_long_term_results.png"
    )

    # Transformer Short-term prediction
    plot_results(
        "transformer_final_predictions_output96.npy",
        "transformer_final_ground_truths_output96.npy",
        title="Transformer Short-Term Prediction (96 Hours)",
        output_file="transformer_short_term_results.png"
    )

    # Transformer Long-term prediction
    plot_results(
        "transformer_final_predictions_output240.npy",
        "transformer_final_ground_truths_output240.npy",
        title="Transformer Long-Term Prediction (240 Hours)",
        output_file="transformer_long_term_results.png"
    )

    # TFT Short-term prediction
    plot_results(
        "tft_final_predictions_output96.npy",
        "tft_final_ground_truths_output96.npy",
        title="TFT Short-Term Prediction (96 Hours)",
        output_file="tft_short_term_results.png"
    )

    # TFT Long-term prediction
    plot_results(
        "tft_final_predictions_output240.npy",
        "tft_final_ground_truths_output240.npy",
        title="TFT Long-Term Prediction (240 Hours)",
        output_file="tft_long_term_results.png"
    )

    # Autoformer Short-term prediction
    plot_results(
        "autoformer_final_predictions_output96.npy",
        "autoformer_final_ground_truths_output96.npy",
        title="Autoformer Short-Term Prediction (96 Hours)",
        output_file="autoformer_short_term_results.png"
    )

    # Autoformer Long-term prediction
    plot_results(
        "autoformer_final_predictions_output240.npy",
        "autoformer_final_ground_truths_output240.npy",
        title="Autoformer Long-Term Prediction (240 Hours)",
        output_file="autoformer_long_term_results.png"
    )
