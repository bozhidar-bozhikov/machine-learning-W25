import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms

from binary_classifier import BinaryImageClassifier
from plots import parse_log_file, create_plots


# Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "cats_vs_dogs_split")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
INPUT_SIZE = 224
WEIGHT_DECAY = 1e-4 #L2 regularization
ES_PATIENCE = 5  #early stopping
LR_PATIENCE = 3  #LR scheduler if no improvement for N epochs
LR_FACTOR = 0.5  #multiply LR by N when reducing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS_DIR = os.path.join(BASE_DIR, "saved_runs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

#determine run number automatically
existing_runs = [f for f in os.listdir(RUNS_DIR) if f.startswith("run") and f.endswith(".pth")]
run_number = len(existing_runs) + 1

#setup logging
log_path = os.path.join(LOGS_DIR, f"log{run_number}.txt")
log_file = open(log_path, 'w')

def log_print(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

#log hyperparameters
log_print(f"BATCH_SIZE = {BATCH_SIZE}")
log_print(f"EPOCHS = {EPOCHS}")
log_print(f"LR = {LR}")
log_print(f"WEIGHT_DECAY = {WEIGHT_DECAY}")
log_print(f"ES_PATIENCE = {ES_PATIENCE}")
log_print(f"LR_PATIENCE = {LR_PATIENCE}")
log_print(f"LR_FACTOR = {LR_FACTOR}")
log_print(f"PyTorch version: {torch.__version__}")
log_print(f"Using device: {DEVICE}")

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
#optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',  #for validation accuracy
    factor=LR_FACTOR,  #multiply LR by this factor
    patience=LR_PATIENCE  #wait this many epochs before reducing
)

# Training Loop

best_val_acc = 0.0
#for early stopping
patience_counter = 0

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

    current_lr = optimizer.param_groups[0]['lr']

    log_print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} "
        f"Learning rate: {current_lr:.6f} "
        f"ES Patience: {patience_counter}/{ES_PATIENCE}"
    )

    # Save best model for this run

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0 
        
        run_path = os.path.join(RUNS_DIR, f"run{run_number}.pth")
        torch.save(model.state_dict(), run_path)
        best_path = os.path.join(RUNS_DIR, "best_run.pth")
        torch.save(model.state_dict(), best_path)
        log_print(f"Saved best model for run{run_number} with val acc: {best_val_acc:.4f}")
    else:
        patience_counter += 1
        log_print(f"No improvement. Early stopping patience: {patience_counter}/{ES_PATIENCE}")
        
        if patience_counter >= ES_PATIENCE:
            log_print(f"Early stopping triggered at epoch {epoch+1}")
            break

    #lr scheduling
    old_lr = current_lr
    scheduler.step(val_acc)
    new_lr = scheduler.get_last_lr()[0]

    if new_lr < old_lr:
        log_print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

log_print("Training finished")
log_print(f"Best Val Acc for this run: {best_val_acc:.4f}")

try:
    epochs, train_losses, train_accs, val_losses, val_accs, hyperparams = parse_log_file(log_path)
    loss_plot, acc_plot = create_plots(
        epochs, train_losses, train_accs, val_losses, val_accs,
        run_number, hyperparams, PLOTS_DIR
    )
except Exception as e:
    print(f"Error generating plots: {e}")