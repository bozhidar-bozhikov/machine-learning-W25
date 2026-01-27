import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from binary_classifier import BinaryImageClassifier


# Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "cats_vs_dogs_split")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS_DIR = os.path.join(BASE_DIR, "saved_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# Transforms

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])

# Datasets & Loaders

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model, Loss, Optimizer

model = BinaryImageClassifier(input_channels=3, input_size=INPUT_SIZE).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# Training Loop

best_val_acc = 0.0

# Determine run number automatically
existing_runs = [f for f in os.listdir(RUNS_DIR) if f.startswith("run") and f.endswith(".pth")]
run_number = len(existing_runs) + 1

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds = (outputs >= 0.5).long()
        running_corrects += torch.sum(preds == labels.long())

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)


    # Validation

    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            preds = (outputs >= 0.5).long()
            val_corrects += torch.sum(preds == labels.long())

    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
    )

    # Save best model for this run

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        # Save best for this run
        run_path = os.path.join(RUNS_DIR, f"run{run_number}.pth")
        torch.save(model.state_dict(), run_path)

        # Also save the best overall
        best_path = os.path.join(RUNS_DIR, "best_run.pth")
        torch.save(model.state_dict(), best_path)

        print(f"Saved best model for run{run_number} with val acc: {best_val_acc:.4f}")

print("Training finished")
print(f"Best Val Acc for this run: {best_val_acc:.4f}")

