import os
import csv
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class BoneCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['normal', 'cancer']
        self.data = []
        
        # Parse Kaggle / Roboflow CSV annotations
        # Format: filename, cancer, normal
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if not row: continue # Skip empty rows
                filename = row[0].strip()
                # row[1] corresponds to the 'cancer' column
                # row[2] corresponds to the 'normal' column
                try:
                    is_cancer = int(row[1].strip())
                    label = 1 if is_cancer == 1 else 0
                    self.data.append((filename, label))
                except ValueError:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image (convert to RGB as some might be grayscale)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    train_csv = os.path.join(train_dir, '_classes.csv')
    valid_csv = os.path.join(valid_dir, '_classes.csv')
    test_csv = os.path.join(test_dir, '_classes.csv')

    # Define transforms for Transfer Learning (ResNet50 uses 224x224)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Use the custom dataset to parse Kaggle _classes.csv annotations
    train_data = BoneCancerDataset(csv_file=train_csv, img_dir=train_dir, transform=train_transforms)
    
    if os.path.exists(valid_csv):
        valid_data = BoneCancerDataset(csv_file=valid_csv, img_dir=valid_dir, transform=val_test_transforms)
    else:
        valid_data = BoneCancerDataset(csv_file=test_csv, img_dir=test_dir, transform=val_test_transforms)

    test_data = BoneCancerDataset(csv_file=test_csv, img_dir=test_dir, transform=val_test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class_names = train_data.classes

    return train_loader, valid_loader, test_loader, class_names

if __name__ == "__main__":
    # Test loader
    data_dir = r"C:\Users\rabia\OneDrive\Desktop\archive"
    train_loader, valid_loader, test_loader, class_names = get_data_loaders(data_dir)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
