## Model Architecture

The CNN architecture is specifically designed to be lightweight while maintaining high accuracy:

1. Convolutional Layers:
   - First layer: 1 → 6 channels (3x3 kernel)
   - Second layer: 6 → 12 channels (3x3 kernel)
   - Each followed by ReLU and MaxPool2d

2. Fully Connected Layers:
   - Hidden layer: 588 → 24 neurons
   - Output layer: 24 → 10 neurons (one for each digit)

Total parameters: ~15000