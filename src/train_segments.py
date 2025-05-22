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
print(PROJECT_ROOT)
print(SRC_PATH)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import get_dataloaders, train_model, plot_training_curves

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

COUNTRIES = ["Albania","Andorra","Argentina","Australia","Austria","Bangladesh","Belgium","Bhutan","Bolivia","Botswana","Brazil","Bulgaria","Cambodia","Canada","Chile","Colombia","Croatia","Czechia","Denmark","Dominican Republic","Ecuador","Estonia","Eswatini","Finland","France","Germany","Ghana","Greece","Greenland","Guatemala","Hungary","Iceland","Indonesia","Ireland","Israel","Italy","Japan","Jordan","Kenya","Kyrgyzstan","Latvia","Lesotho","Lithuania","Luxembourg","Malaysia","Mexico","Mongolia","Montenegro","Netherlands","New Zealand","Nigeria","North Macedonia","Norway","Palestine","Peru","Philippines","Poland","Portugal","Romania","Russia","Senegal","Serbia","Singapore","Slovakia","Slovenia","South Africa","South Korea","Spain","Sri Lanka","Sweden","Switzerland","Taiwan","Thailand","Turkey","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay"]
num_classes = len(COUNTRIES)

print("Number of classes:", num_classes)

project_root = Path().resolve().parent
SEGMENTS = ['road', 'vegetation', 'terrain']

for segment in SEGMENTS:
    print(f"\nTraining model for {segment}")
    
    train_root = project_root / "datasets" / "segmented" / segment / "final_datasets" / "train"
    dev_root = project_root / "datasets" / "segmented" / segment / "final_datasets" / "val"

    print("Loading data...")
    print(train_root)
    print(dev_root)

    train_loader = get_dataloaders(train_root, batch_size=32)
    val_loader = get_dataloaders(dev_root, batch_size=32)

    print("Data loaded")

    print("Loading model...")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    for param in model.parameters():
        param.data = param.data.to(device)
    for buf in model.buffers():
        buf.data = buf.data.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training for {segment}...")

    model_save_path = project_root / "models" / f"resnet_finetuned_{segment}" / "main.pth"
    results = train_model(model, train_loader, val_loader, device, optimizer, 
                         model_save_path, criterion=nn.CrossEntropyLoss(), 
                         epochs=50, eval_every=50, patience=3)

    plot_training_curves(PROJECT_ROOT, results["train_loss"], results["train_acc"],
                        results["val_loss"], results["val_acc"],
                        eval_every=50, segment=segment)