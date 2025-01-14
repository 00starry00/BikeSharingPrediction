import argparse
from train import train_model

def main():
    parser = argparse.ArgumentParser(description="Bike Sharing Demand Prediction")
    parser.add_argument("--model", type=str, default="LSTM", choices=["LSTM", "Transformer", "TFT", "Autoformer"], help="Model to use for training")
    parser.add_argument("--train_path", type=str, default="data/train_data.csv", help="Path to training data")
    parser.add_argument("--test_path", type=str, default="data/test_data.csv", help="Path to testing data")
    parser.add_argument("--input_window", type=int, default=96, help="Input window size")
    parser.add_argument("--output_window", type=int, default=96, help="Output window size")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads (Transformer only)")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="Feedforward dimension (Transformer only)")

    args = parser.parse_args()
    

    train_model(
        model_name=args.model,
        train_path=args.train_path,
        test_path=args.test_path,
        input_window=args.input_window,
        output_window=args.output_window,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_heads=args.num_heads,  # Corrected typo here
        dim_feedforward=args.dim_feedforward  # Corrected typo here
    )

if __name__ == "__main__":
    main()
