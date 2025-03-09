import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_resnet(data_path, weights_path, num_epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    """
    Trains the ResNet model.

    Args:
        data_path (str): Path to training data.
        weights_path (str): Path to save trained weights.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').
    """
    device = torch.device(device)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    train_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = models.resnet50(pretrained=False)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Save the trained model weights
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

if __name__ == "__main__":
    data_path = "/data/annotated"
    weights_path = "resnet50_weights.pth"
    train_resnet(data_path, weights_path, num_epochs=10, batch_size=32, learning_rate=0.001, device='cuda')