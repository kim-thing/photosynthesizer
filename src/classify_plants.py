# src/classify_plants.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import sys
import os

# Make sure src/ can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Constants ===
NUM_LABELS = 102  # Because Oxford dataset has 102 flower classes
NUM_EPOCHS = 20   # 🚀 Train longer for better accuracy
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# === Model ===
class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_LABELS),
            nn.Sigmoid()  # Multi-label output
        )

    def forward(self, x):
        return self.base_model(x)

# === Training function (optional) ===
def train_classifier(data_dir, num_epochs=NUM_EPOCHS):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PlantClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in dataloader:
            labels = labels.type(torch.FloatTensor)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "models/plant_classifier.pt")
    print("✅ Model saved to models/plant_classifier.pt")
    return model

# === Main Run Block ===
if __name__ == "__main__":
    from src.prepare_data import prepare_dataset
    from src.dataset import FlowerDataset
    from torch.utils.data import DataLoader

    os.makedirs("models", exist_ok=True)

    # Step 1: Load labels and splits
    labels, train_idx, val_idx, test_idx = prepare_dataset()

    # Step 2: Dataset + DataLoader
    train_dataset = FlowerDataset("data/flowers/jpg", train_idx, labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Step 3: Init model, loss, optimizer
    model = PlantClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 4: Training
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=NUM_LABELS).float()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

    # Step 5: Save model
    torch.save(model.state_dict(), "models/plant_classifier.pt")
    print("✅ Model saved to models/plant_classifier.pt")
