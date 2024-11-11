import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import multiprocessing
import sys
import os

print("Torch version - ", torch.__version__)
multiprocessing.set_start_method('spawn', force=True)

# Define the CNN model
class MediumCNN(nn.Module):
    def __init__(self):
        super(MediumCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)  # Adjust the input size for flattened output
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        # Pass through first convolutional block
        x = self.pool1(self.relu(self.conv2(self.relu(self.conv1(x)))))
        
        # Pass through second convolutional block
        x = self.pool2(self.relu(self.conv4(self.relu(self.conv3(x)))))
        
        # Pass through third convolutional block
        x = self.pool3(self.relu(self.conv6(self.relu(self.conv5(x)))))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 512 * 4 * 4)
        
        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def loadDataset(batch_size):
    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = MediumCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Function to benchmark training
def train_and_benchmark(epochs, batchSize, checkpoint_dir="checkpoints"):
    print("Number of Epochs - ", epochs)
    print("Batch size - ", batchSize)

    model = MediumCNN().to(device)
    loadDataset(batchSize)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load CIFAR-10 dataset without multiprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)

    start_time = time.time()
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0  # Track the number of correct predictions
        total_samples = 0  # Track the total number of samples

        epoch_start_time = time.time()

        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute predictions and update accuracy counters
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Print loss statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Calculate and print accuracy for the epoch
        epoch_accuracy = 100 * total_correct / total_samples
        print(f"Epoch {epoch + 1} accuracy: {epoch_accuracy:.2f}%")

        # Save checkpoint at the end of the epoch
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.2f} seconds over {epochs} epochs")

def main():
    args = sys.argv[0:]
    print(args)
    batchSize = 128
    epochs = 40
    if len(args) >= 2:
        epochs = int(args[1])
    
    if len(args) >=3:
        batchSize = int(args[2])
    
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # Start method can only be set once per session, ignore if already set.

    # Run the benchmark
    train_and_benchmark(epochs, batchSize, "checkpoints")

if __name__ == "__main__":
    main()
