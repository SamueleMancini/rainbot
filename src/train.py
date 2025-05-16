import sys, os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Compute absolute path to the `src/` folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import get_dataloaders, train_model, plot_training_curves

# Set device (MPS for Apple Silicon, CPU otherwise)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# List of countries for classification
COUNTRIES = ["Albania","Andorra","Argentina","Australia","Austria","Bangladesh","Belgium","Bhutan","Bolivia","Botswana","Brazil","Bulgaria","Cambodia","Canada","Chile","Colombia","Croatia","Czechia","Denmark","Dominican Republic","Ecuador","Estonia","Eswatini","Finland","France","Germany","Ghana","Greece","Greenland","Guatemala","Hungary","Iceland","Indonesia","Ireland","Israel","Italy","Japan","Jordan","Kenya","Kyrgyzstan","Latvia","Lesotho","Lithuania","Luxembourg","Malaysia","Mexico","Mongolia","Montenegro","Netherlands","New Zealand","Nigeria","North Macedonia","Norway","Palestine","Peru","Philippines","Poland","Portugal","Romania","Russia","Senegal","Serbia","Singapore","Slovakia","Slovenia","South Africa","South Korea","Spain","Sri Lanka","Sweden","Switzerland","Taiwan","Thailand","Turkey","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay"]
num_classes = len(COUNTRIES)

# Set up data paths
project_root = Path().resolve().parent
train_root = project_root / "datasets" / "final_datasets" / "train"
test_root = project_root / "datasets" / "final_datasets" / "test"

# Create data loaders
train_loader = get_dataloaders(train_root, batch_size=32)
test_loader = get_dataloaders(test_root, batch_size=32)

# Initialize ResNet50 model with pretrained weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Modify final layer for our number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Move all parameters and buffers to device
for param in model.parameters():
    param.data = param.data.to(device)
for buf in model.buffers():
    buf.data = buf.data.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,  # Using test set for validation
    device=device,
    optimizer=optimizer,
    save_path=project_root / "models" / "resnet_finetuned" / "model.pth",
    criterion=criterion,
    epochs=10,
    eval_every=50,
    patience=3
)

# Plot training results
plot_training_curves(
    results["train_losses"],
    results["val_losses"],
    results["train_accs"],
    results["val_accs"]
)
