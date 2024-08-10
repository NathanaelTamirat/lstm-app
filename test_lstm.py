import torch
from typing import Optional, Tuple
from lstm import LSTM  

def test_lstm():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define hyperparameters
    input_size = 10
    hidden_size = 20
    num_layers = 2
    sequence_length = 5
    batch_size = 3

    # Initialize the LSTM model
    lstm = LSTM(input_size, hidden_size, num_layers)

    # Create random input data
    x = torch.randn(sequence_length, batch_size, input_size)

    # Forward pass without initial state
    output, (h, c) = lstm(x)

    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {h.shape}")
    print(f"Cell state shape: {c.shape}")

    # Forward pass with initial state
    initial_h = torch.randn(num_layers, batch_size, hidden_size)
    initial_c = torch.randn(num_layers, batch_size, hidden_size)
    output, (h, c) = lstm(x, (initial_h, initial_c))

    print("\nWith initial state:")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {h.shape}")
    print(f"Cell state shape: {c.shape}")

if __name__ == "__main__":
    test_lstm()