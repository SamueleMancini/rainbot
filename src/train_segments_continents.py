
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

# Setup paths
PROJECT_ROOT = Path().resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from utils import get_dataloaders_continents, train_model, plot_training_curves

# Device config
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Define labels as continents
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
num_classes = len(CONTINENTS)
print("Number of classes (continents):", num_classes)

# Define valid segments
SEGMENTS_DIR = PROJECT_ROOT / "datasets" / "segmented"
SEGMENTS = os.listdir(SEGMENTS_DIR)

# Train model for each segment
for segment in SEGMENTS:
    if os.path.exists(PROJECT_ROOT / "models" / f"resnet_finetuned_{segment}_continents" / "main.pth"):
        print(f"\n🔧 Skipping {segment} because it already exists")
        continue
    print(f"\n🔧 Training model for: {segment}")
    
    train_root = SEGMENTS_DIR / segment / "final_datasets_continents" / "train"
    val_root   = SEGMENTS_DIR / segment / "final_datasets_continents" / "val"

    print("📦 Loading data from:")
    print(f"  Train: {train_root}")
    print(f"  Val:   {val_root}")
    try:
        train_loader = get_dataloaders_continents(train_root, batch_size=32)
        val_loader   = get_dataloaders_continents(val_root, batch_size=32)
    except Exception as e:
        print(f"Error loading data: {e}")
        continue

    print("✅ Data loaded")

    # Load and modify model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    for param in model.parameters():
        param.data = param.data.to(device)
    for buf in model.buffers():
        buf.data = buf.data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Starting training for {segment}...")

    model_save_path = PROJECT_ROOT / "models" / f"resnet_finetuned_{segment}_continents" / "main.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    results = train_model(
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        model_save_path,
        criterion=criterion,
        epochs=50,
        eval_every=50,
        patience=3
    )

    plot_training_curves(
        PROJECT_ROOT,
        results["train_loss"],
        results["train_acc"],
        results["val_loss"],
        results["val_acc"],
        eval_every=50,
        segment=segment
    )

print("\n✅ All continent-based models trained.")
