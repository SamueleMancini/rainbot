import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import (
    top_k_accuracy_score,
    classification_report,
    confusion_matrix
)
import torch.nn.functional as F
from transformers import SwinForImageClassification, SwinConfig

############################## Data Loading and Preprocessing ##############################

COUNTRIES = ['Albania',
 'Andorra',
 'Australia',
 'Austria',
 'Bangladesh',
 'Belgium',
 'Bhutan',
 'Bolivia',
 'Brazil',
 'Bulgaria',
 'Cambodia',
 'Canada',
 'Chile',
 'Colombia',
 'Croatia',
 'Czechia',
 'Denmark',
 'Dominican Republic',
 'Ecuador',
 'Estonia',
 'Eswatini',
 'Finland',
 'France',
 'Germany',
 'Greece',
 'Guatemala',
 'Hungary',
 'Iceland',
 'Indonesia',
 'Ireland',
 'Israel',
 'Italy',
 'Japan',
 'Jordan',
 'Latvia',
 'Lesotho',
 'Lithuania',
 'Luxembourg',
 'Malaysia',
 'Mexico',
 'Montenegro',
 'Netherlands',
 'New Zealand',
 'North Macedonia',
 'Norway',
 'Palestine',
 'Peru',
 'Poland',
 'Portugal',
 'Romania',
 'Russia',
 'Serbia',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'South Africa',
 'South Korea',
 'Spain',
 'Sweden',
 'Switzerland',
 'Taiwan',
 'Thailand',
 'Turkey',
 'United Arab Emirates',
 'United Kingdom',
 'United States']
num_classes = len(COUNTRIES)
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
num_classes_continents = len(CONTINENTS)


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
    
class ContinentImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for idx, continent in enumerate(CONTINENTS):
            continent_dir = root_dir / continent
            for img_file in continent_dir.iterdir():
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

def get_dataloaders_continents(root_dir, batch_size=32):
    dataset = ContinentImageDataset(root_dir, transform)
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


def plot_training_curves(project_root, train_losses, train_accs, val_losses, val_accs, eval_every=50, segment=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    project_root = Path(project_root)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
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

    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    if segment:
        plot_path = plots_dir / f"training_curves_{segment}.png"
    else:
        plot_path = plots_dir / "training_curves.png"

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()




############################## Evaluation ##############################


def load_model(model_path, device, num_classes=num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_vit_model(model_path, device, num_classes=num_classes):
    """
    model_path   : Path or str to folder containing config.json & pytorch_model.bin
    device      : torch.device
    num_classes : int, number of output labels
    id2label    : dict mapping label IDs to string names
    label2id    : dict mapping string names to label IDs
    """


    label2id = {label: idx for idx, label in enumerate(COUNTRIES)}
    id2label = {idx: label for label, idx in label2id.items()}

    config = SwinConfig.from_pretrained(model_path)
    config.num_labels = num_classes
    config.id2label   = id2label
    config.label2id   = label2id

    model = SwinForImageClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    model.to(device).eval()
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
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        zero_division=0
    )
    return top3, top5, report


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


def plot_random_image_with_label_and_prediction(test_root, model, device):
    # Pick random image and true label
    all_countries = [d for d in test_root.iterdir() if d.is_dir()]
    country = random.choice(all_countries).name
    img_files = list((test_root / country).glob("*.jpg"))
    img_path = random.choice(img_files)

    img = Image.open(img_path).convert("RGB")

    # Preprocess and predict
    input_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        pred_label = COUNTRIES[pred_idx]

    # Visualize the image with true label
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{country}, Pred: {pred_label}")
    plt.show()



############################## GradCAM ##############################


def gradCAM(test_root, model, device):
    """
    test_root: folder of per-country subfolders (test set)
    model: your fine-tuned ResNet model
    device: torch.device("mps" or "cuda" or "cpu")
    """
    # 0) Hook containers
    activations = gradients = None

    # 1) Hook callbacks
    def forward_hook(module, inp, outp):
        nonlocal activations
        activations = outp.detach()
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # 2) Attach hooks to the last ResNet conv layer
    target_layer = model.layer4[-1].conv3
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # 3) Pick a random test image + ground truth
    country_dirs = [d for d in test_root.iterdir() if d.is_dir()]
    true_country = random.choice(country_dirs).name
    img_path = random.choice(list((test_root/true_country).glob("*.jpg")))
    orig = Image.open(img_path).convert("RGB")
    # Keep a display copy
    disp = orig.resize((224,224), Image.BILINEAR)

    # 4) Preprocess and forward for classification (no no_grad!)
    model.to(device).eval()
    inp = transform(orig).unsqueeze(0).to(device)

    # Zero grads before forward
    model.zero_grad()
    outputs = model(inp)               # <-- no torch.no_grad() here!
    pred_idx = outputs.argmax(dim=1).item()
    pred_label = COUNTRIES[pred_idx]

    # 5) Backward on the predicted class score
    score = outputs[0, pred_idx]
    score.backward()

    # 6) Build Grad-CAM
    weights = gradients.mean(dim=(2,3), keepdim=True)    # (1, C, 1, 1)
    cam_map = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam_map = F.interpolate(cam_map,
                            size=inp.shape[2:],
                            mode='bilinear',
                            align_corners=False)
    cam = cam_map.squeeze().cpu().numpy()
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)

    # 7) Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))

    # Original + labels
    ax1.imshow(disp)
    ax1.set_title(f"True: {true_country}\nPred: {pred_label}")
    ax1.axis('off')

    # Grad-CAM overlay
    img_np = np.array(disp)/255.0
    heatmap = plt.cm.jet(cam)[...,:3]
    overlay = 0.4*heatmap + 0.6*img_np
    ax2.imshow(overlay)
    ax2.set_title("Grad-CAM")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # 8) Remove hooks
    handle_f.remove()
    handle_b.remove()
