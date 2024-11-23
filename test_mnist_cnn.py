import unittest
import torch
from mnist_cnn import LightweightMNISTCNN, train_model
from unittest.case import TestCase
import os
import platform
from functools import wraps
import time

def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if platform.system() == 'Windows':
                # Simple timeout implementation for Windows
                start_time = time.time()
                result = func(*args, **kwargs)
                if time.time() - start_time > seconds:
                    raise TimeoutError(error_message)
                return result
            else:
                # Unix-based systems can use SIGALRM
                import signal
                def _handle_timeout(signum, frame):
                    raise TimeoutError(error_message)

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
        return wrapper
    return decorator

class TestMNISTCNN(unittest.TestCase):
    def setUp(self):
        self.model = LightweightMNISTCNN()
        
    def test_parameter_count(self):
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertLess(param_count, 25000, 
            f"Model has {param_count} parameters, which exceeds the limit of 25,000")
        print(f"\nParameter count test passed. Total parameters: {param_count}")

    @timeout(300)  # 5 minutes timeout
    def test_model_accuracy(self):
        # Train the model and get accuracy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LightweightMNISTCNN().to(device)
        
        # Redirect stdout to capture prints during training
        import sys
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            train_model()
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Extract accuracy from the output
        accuracy_line = [line for line in output.split('\n') if 'Test Accuracy' in line][0]
        accuracy = float(accuracy_line.split(':')[1].strip('%'))
        
        self.assertGreater(accuracy, 95.0, 
            f"Model accuracy {accuracy:.2f}% is below the required 95%")
        print(f"\nAccuracy test passed. Model achieved {accuracy:.2f}% accuracy")

if __name__ == '__main__':
    unittest.main()
