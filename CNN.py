import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATASET_PATH = r"C:\Users\DELL\Desktop\CNN-Myopia1\PALM_Dataset\Training\Images"
LABELS_PATH = r"C:\Users\DELL\Desktop\CNN-Myopia1\PALM_Dataset\Training\Labels.csv"

# Read CSV file
df = pd.read_csv(LABELS_PATH, header=None, names=["imgName", "Label"])

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset
class MyopiaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):  
            print(f"Warning: {img_path} not found. Skipping...")
            return None  # Return None if image is missing

        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(float(self.dataframe.iloc[idx, 1]), dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label

# Split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create Datasets and DataLoaders
train_dataset = MyopiaDataset(train_df, DATASET_PATH, transform=transform)
val_dataset = MyopiaDataset(val_df, DATASET_PATH, transform=transform)
test_dataset = MyopiaDataset(test_df, DATASET_PATH, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN Model
class MyopiaClassifier(nn.Module):
    def __init__(self):
        super(MyopiaClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x  # Output raw logits

# Initialize model
model = MyopiaClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead of BCELoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        if batch is None:
            continue  # Skip None values from missing images
        images, labels = batch
        images, labels = images.to(device), labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Save model
torch.save(model.state_dict(), "myopia_classifier.pth")
