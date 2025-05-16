import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    top_k_accuracy_score,
    classification_report,
    confusion_matrix
)

############################## Data Loading and Preprocessing ##############################

COUNTRIES = ["Albania","Andorra","Argentina","Australia","Austria","Bangladesh","Belgium","Bhutan","Bolivia","Botswana","Brazil","Bulgaria","Cambodia","Canada","Chile","Colombia","Croatia","Czechia","Denmark","Dominican Republic","Ecuador","Estonia","Eswatini","Finland","France","Germany","Ghana","Greece","Greenland","Guatemala","Hungary","Iceland","Indonesia","Ireland","Israel","Italy","Japan","Jordan","Kenya","Kyrgyzstan","Latvia","Lesotho","Lithuania","Luxembourg","Malaysia","Mexico","Mongolia","Montenegro","Netherlands","New Zealand","Nigeria","North Macedonia","Norway","Palestine","Peru","Philippines","Poland","Portugal","Romania","Russia","Senegal","Serbia","Singapore","Slovakia","Slovenia","South Africa","South Korea","Spain","Sri Lanka","Sweden","Switzerland","Taiwan","Thailand","Turkey","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay"]
num_classes = len(COUNTRIES)



class CountryImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for idx, country in enumerate(COUNTRIES):
            country_dir = root_dir / country
            for img_file in country_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_file, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def get_dataloaders(root_dir, batch_size=32):
    dataset = CountryImageDataset(root_dir, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader



############################## Model ##############################

def make_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model



############################## Training ##############################

def compute_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds==labels).sum().item() / labels.size(0)


def compute_train_accuracy(model, train_loader, device):
    """Returns average accuracy over the entire train_loader (no grad)."""
    model.eval()
    total_acc, total_count = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            total_acc += compute_accuracy(outs, lbls) * imgs.size(0)
            total_count += imgs.size(0)
    return total_acc / total_count


def compute_val_metrics(model, val_loader, criterion, device):
    """Returns (avg_loss, avg_acc) over the entire val_loader (no grad)."""
    model.eval()
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            total_loss += criterion(outs, lbls).item() * imgs.size(0)
            total_acc  += compute_accuracy(outs, lbls) * imgs.size(0)
            total_count += imgs.size(0)
    avg_loss = total_loss / total_count
    avg_acc  = total_acc  / total_count
    return avg_loss, avg_acc



def train_model(model, train_loader, val_loader, device, optimizer, checkpoint_path, criterion=nn.CrossEntropyLoss(), epochs=10, eval_every=50, patience=3):
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, train_accs = [], []
    val_losses, val_accs     = [], []

    batches = len(train_loader)

    batch = 1
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outs = model(imgs)
                loss = criterion(outs, lbls)
                loss.backward()
                optimizer.step()

                # Log per‐batch
                acc = compute_accuracy(outs, lbls)
                train_losses.append(loss.item())
                train_accs.append(acc)

                if batch % eval_every == 0:
                    # evaluate on validation set
                    vl, va = compute_val_metrics(model, val_loader, criterion, device)
                    val_losses.append(vl)
                    val_accs.append(va)

                    print(f"[Epoch {epoch}/{epochs}, Batch {batch}/{batches}] "
                          f"Train Loss={loss:.4f}, Train Acc={acc:.4f} | "
                          f"Val Loss={vl:.4f}, Val Acc={va:.4f}")

                    # Early stopping + checkpoint
                    if vl < best_val_loss:
                        best_val_loss = vl
                        patience_counter = 0
                        torch.save(model.state_dict(), checkpoint_path)
                        print("  ↳ Checkpoint saved.")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print("Early stopping triggered.")
                            raise KeyboardInterrupt

                    model.train()

                batch += 1

            # End‐of‐epoch summary
            vl, va = compute_val_metrics(model, val_loader, criterion, device)
            train_acc = compute_train_accuracy(model, train_loader, device)
            last_loss = train_losses[-1]
            last_acc  = train_accs[-1]

            print(f"[Epoch {epoch}/{epochs}] "
                  f"Last Batch Train Loss={last_loss:.4f}, Last Batch Train Acc={last_acc:.4f} | "
                  f"Val Loss={vl:.4f}, Val Acc={va:.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving latest model…")
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete.")
    return {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs
    }


def plot_training_curves(project_root, train_losses, train_accs, val_losses, val_accs, eval_every=50):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # X for val losses/accs corresponds to evaluation steps
    x_val = np.arange(eval_every, eval_every * len(val_losses) + 1, eval_every)

    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(x_val, val_losses, label='Val Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(train_accs, label='Train Acc')
    axs[1].plot(x_val, val_accs, label='Val Acc')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Save the plot
    plt.savefig(plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

    ############################## Evaluation ##############################

def load_model(model_path, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 79)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


softmax = torch.nn.Softmax(dim=1)

def evaluate_model(model, data_loader, criterion, device):
    """
    Runs model on data_loader and returns:
      - avg_loss: float
      - top1_acc: float
      - all_targets: np.array shape (N,)
      - all_preds:   np.array shape (N,)
      - all_probs:   np.array shape (N, num_classes)
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            # accumulate loss & top‐1 accuracy
            batch_size = imgs.size(0)
            total_loss    += loss.item() * batch_size
            preds         = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            # store for detailed metrics
            all_probs.append(softmax(outputs).cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # flatten
    all_probs   = np.vstack(all_probs)
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / total_samples
    top1_acc = total_correct / total_samples

    return avg_loss, top1_acc, all_targets, all_preds, all_probs


def print_metrics(all_targets, all_preds, all_probs, class_names):
    """
    Given flattened targets, preds, and probs:
      - prints Top-3/5 accuracy
      - prints classification report
      - plots normalized confusion matrix
    """
    top3 = top_k_accuracy_score(all_targets, all_probs, k=3)
    top5 = top_k_accuracy_score(all_targets, all_probs, k=5)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}\n")

    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        zero_division=0
    )
    print("Classification Report:\n")
    print(report)


def plot_confusion_matrix(all_targets, all_preds, class_names, size=12):
    
    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(size,size))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def show_sample_predictions(all_targets, all_probs, class_names, n=5):
    """
    Prints n random examples of true label vs top-3 predictions+probs.
    """
    total = len(all_targets)
    print(f"\nSample predictions ({n} examples):\n")
    idxs = np.random.choice(total, size=n, replace=False)
    for i in idxs:
        true_lbl = class_names[all_targets[i]]
        probs_i  = all_probs[i]
        topk     = probs_i.argsort()[::-1][:3]
        topk_str = ", ".join(f"{class_names[k]} ({probs_i[k]:.2f})" for k in topk)
        print(f"True: {true_lbl:20s}  ↔  Pred Top-3: {topk_str}")