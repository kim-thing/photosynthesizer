import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights


# Modify this based on our label set
NUM_LABELS = 102  # not sure if it should be 5 or 102

class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_LABELS),
            nn.Sigmoid()  # Multi-label output IMPORTANT
        )

    def forward(self, x):
        return self.base_model(x)


def train_classifier(data_dir, num_epochs=5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PlantClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in dataloader:
            labels = labels.type(torch.FloatTensor)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "models/plant_classifier.pt")
    return model


if __name__ == "__main__":
    from src.prepare_data import prepare_dataset
    from src.dataset import FlowerDataset
    from torch.utils.data import DataLoader
    import os

    os.makedirs("models", exist_ok=True)  # ensureS ouur models/ exists

    # Step 1: Get labels and split indices
    labels, train_idx, val_idx, test_idx = prepare_dataset()

    # Step 2: Create dataset + dataloader using our custom dataset class (need to check if i did it right)
    train_dataset = FlowerDataset("data/flowers/jpg", train_idx, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Step 3: Initialize model 'criterion" optimizer
    model = PlantClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 4: Training loop
    for epoch in range(5):
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

        print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(train_loader):.4f}")

    # Step 5: This saves the model, dude making it create a folder too so long
    torch.save(model.state_dict(), "models/plant_classifier.pt")
    print("✅ Model saved to models/plant_classifier.pt")

