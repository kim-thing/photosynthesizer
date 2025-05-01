

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import sys
import os
from sklearn.metrics import accuracy_score, f1_score

#  imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#  cconstants 
NUM_LABELS = 102
NUM_EPOCHS = 10  
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

#  model 
class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),




            nn.Dropout(0.4),
            nn.Linear(128, NUM_LABELS)
            # replaced Sigmoids,didnt make sense for 1 layer
        )

    def forward(self, x):
        return self.base_model(x)

# Train Function 
def train_classifier(train_loader, val_loader):
    model = PlantClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #had
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# attempt at early Stopping 
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5  # we cna chnage this but it takes really long to run
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        train_preds = []
        train_labels = []



        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()





            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_preds.append(preds.detach().cpu())
            train_labels.append(labels.detach().cpu())

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)

        train_acc = accuracy_score(train_labels.numpy(), train_preds.numpy())
        train_f1 = f1_score(train_labels.numpy(), train_preds.numpy(), average='macro')

#  Validation Phase 
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()



                preds = outputs.argmax(dim=1)
                val_preds.append(preds.cpu())
                val_labels_list.append(labels.cpu())
        val_preds = torch.cat(val_preds)
        val_labels_list = torch.cat(val_labels_list)
        val_acc = accuracy_score(val_labels_list.numpy(), val_preds.numpy())
        val_f1 = f1_score(val_labels_list.numpy(), val_preds.numpy(), average='macro')

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} || Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")



        #  Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/plant_classifier.pt")
            print(f"OOOOOO Validation loss improved, model saved!")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stop_patience:
            print(">>> Early stopping ") #lowkey the emojies helped me see the outputs better so i added more 
            break

# main whar we had before tweaked
if __name__ == "__main__":
    from src.prepare_data import prepare_dataset
    from src.dataset import FlowerDataset

    os.makedirs("models", exist_ok=True)

    #  Data Augmentation
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

    # Datasets ref
    train_dataset = FlowerDataset("data/flowers/jpg", train_idx, labels, transform=train_transform)
    val_dataset = FlowerDataset("data/flowers/jpg", val_idx, labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Starting Training
    train_classifier(train_loader, val_loader)
