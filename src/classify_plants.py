# src/classify_plants.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import sys
import os
from sklearn.metrics import accuracy_score, f1_score

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Constants ===
NUM_LABELS = 102
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

# === Model ===
class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout to prevent overfitting
            nn.Linear(128, NUM_LABELS),
            nn.Sigmoid()  # Multilabel
        )

    def forward(self, x):
        return self.base_model(x)

# === Train Function ===
def train_classifier(train_loader, val_loader):
    model = PlantClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)  # Ensure LongTensor

            one_hot_labels = nn.functional.one_hot(labels, num_classes=NUM_LABELS).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = (outputs > 0.5).float()
            train_preds.append(preds.detach().cpu())
            train_labels.append(one_hot_labels.detach().cpu())

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)

        train_acc = accuracy_score(train_labels.numpy().flatten(), train_preds.numpy().flatten())
        train_f1 = f1_score(train_labels.numpy().flatten(), train_preds.numpy().flatten(), zero_division=0)

        # === Validation Phase ===
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                one_hot_labels = nn.functional.one_hot(labels, num_classes=NUM_LABELS).float()

                outputs = model(images)
                loss = criterion(outputs, one_hot_labels)

                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                val_preds.append(preds.cpu())
                val_labels_list.append(one_hot_labels.cpu())

        val_preds = torch.cat(val_preds)
        val_labels_list = torch.cat(val_labels_list)

        val_acc = accuracy_score(val_labels_list.numpy().flatten(), val_preds.numpy().flatten())
        val_f1 = f1_score(val_labels_list.numpy().flatten(), val_preds.numpy().flatten(), zero_division=0)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} || Val Loss: {val_loss/len(val_loader):.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

    torch.save(model.state_dict(), "models/plant_classifier.pt")
    print("✅ Model saved to models/plant_classifier.pt")

# === Main Block ===
if __name__ == "__main__":
    from src.prepare_data import prepare_dataset
    from src.dataset import FlowerDataset

    os.makedirs("models", exist_ok=True)

    # === Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    labels, train_idx, val_idx, test_idx = prepare_dataset()

    # === Datasets
    train_dataset = FlowerDataset("data/flowers/jpg", train_idx, labels, transform=train_transform)
    val_dataset = FlowerDataset("data/flowers/jpg", val_idx, labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # === Start Training
    train_classifier(train_loader, val_loader)
