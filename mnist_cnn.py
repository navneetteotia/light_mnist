import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# Define the CNN architecture
class LightweightMNISTCNN(nn.Module):
    def __init__(self):
        super(LightweightMNISTCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv layer: 1 -> 6 channels, 3x3 kernel
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer: 6 -> 12 channels, 3x3 kernel
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(12 * 7 * 7, 24),
            nn.ReLU(),
            nn.Linear(24, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def show_images(images, title):
    """Display a batch of images."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')

def save_augmentation_examples(dataset, transform, n_samples=8):
    """Save examples of original and augmented images."""
    # Select random samples
    indices = torch.randperm(len(dataset))[:n_samples]
    
    # Get original images and their labels
    original_images = torch.stack([dataset[i][0] for i in indices])
    labels = [dataset[i][1] for i in indices]
    
    # Apply augmentations multiple times to the same images
    augmented_images1 = torch.stack([
        transform(dataset[i][0]) for i in indices
    ])
    augmented_images2 = torch.stack([
        transform(dataset[i][0]) for i in indices
    ])
    
    # Create a single figure
    fig = plt.figure(figsize=(15, 12))
    
    # Plot original images
    ax1 = fig.add_subplot(3, 1, 1)
    plt.title(f'Original Images (Labels: {labels})')
    plt.imshow(np.transpose(make_grid(original_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')
    
    # Plot first set of augmented images
    ax2 = fig.add_subplot(3, 1, 2)
    plt.title('Augmented Version 1')
    plt.imshow(np.transpose(make_grid(augmented_images1, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')
    
    # Plot second set of augmented images
    ax3 = fig.add_subplot(3, 1, 3)
    plt.title('Augmented Version 2')
    plt.imshow(np.transpose(make_grid(augmented_images2, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enhanced data augmentation pipeline
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    base_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )

    # Save augmentation examples before training
    print("Saving augmentation examples...")
    augmentation_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1),
    ])
    save_augmentation_examples(base_dataset, augmentation_transform)
    print("Augmentation examples saved as 'augmentation_examples.png'")

    # Load datasets with full transforms for training
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = LightweightMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count}")

    # Training loop
    print("Starting training...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')


    # Evaluation
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    train_model() 
