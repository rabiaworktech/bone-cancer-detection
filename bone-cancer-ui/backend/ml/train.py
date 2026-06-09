import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_data_loaders
import os

def train_model(data_dir, num_epochs=5, batch_size=32, save_path="bone_cancer_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, valid_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Use more powerful pretrained ResNet50 for transfer learning to increase accuracy
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze lower layers for faster training mapping
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head mapping feature dimensions to our num classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimize only the new head
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    # Adding a learning rate scheduler to improve precision and convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"Starting training for {num_epochs} epoch(s)...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        print(f"Epoch {epoch+1} Summary -> Train Acc: {100.*correct/total:.2f}%, Val Acc: {100.*val_correct/val_total:.2f}%")
    
    print("Training complete. Saving model weights...")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # The dataset path that was provided by the user
    DATA_DIR = r"C:\Users\rabia\OneDrive\Desktop\archive"
    # We train for 1 epoch to quickly generate a working model for the UI
    train_model(DATA_DIR, num_epochs=7, batch_size=64, save_path="bone_cancer_model.pth")
