from transformers import SwinForImageClassification, Trainer, TrainingArguments
from transformers import AutoImageProcessor
from datasets import load_dataset, DatasetDict
from transformers import EarlyStoppingCallback
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import numpy as np
from pathlib import Path

# Your labels
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
label2id = {label: idx for idx, label in enumerate(COUNTRIES)}
id2label = {idx: label for label, idx in label2id.items()}
num_classes = len(COUNTRIES)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# HuggingFace processor (resizes, crops, normalizes)
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")


# Custom transform compatible with HuggingFace
transform = Compose([
    Resize((224, 224)),  # directly match model input
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])


def collate_fn(batch):
    pixel_values = torch.stack([example['pixel_values'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def prepare_dataset(dataset):
    def transform_fn(example):
        image = example["image"].convert("RGB")
        image = transform(image)
        return {"pixel_values": image, "label": label2id[example["label"]]}
    return dataset.map(transform_fn)

# Apply transform safely
def apply_transform(example):
    img = example["image"].convert("RGB")
    tensor = transform(img)
    return {"pixel_values": tensor, "label": example["label"]}

project_root = Path().resolve().parent
SEGMENTS = ['road', 'vegetation', 'terrain']

train_root  = project_root/ "datasets" / "final_datasets" / "train"
dev_root  = project_root/ "datasets" / "final_datasets" / "val"

# Load via datasets.ImageFolder
dataset = DatasetDict({
    "train": load_dataset("imagefolder", data_dir=str(train_root))["train"],
    "val": load_dataset("imagefolder", data_dir=str(dev_root))["train"]
})

# Print class names to verify
print("Detected class labels:", dataset["train"].features["label"].names)

# Apply transform and convert labels
def transform_and_convert(example):
    # Convert image to RGB and apply transform
    img = example["image"].convert("RGB")
    pixel_values = transform(img)
    # Handle both string labels and numeric indices
    label = example["label"]
    if isinstance(label, str):
        label = label2id[label]
    return {
        "pixel_values": pixel_values,
        "label": label
    }

# Apply transformations
dataset = dataset.map(
    transform_and_convert,
    remove_columns=["image"]
)

# Set the format to torch tensors
dataset.set_format(type="torch", columns=["pixel_values", "label"])

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

training_args = TrainingArguments(
    output_dir=str(project_root / "models" / f"swin_b_finetuned_new"),
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=100,
    learning_rate=5e-5,
    logging_dir=str(project_root / "logs" / f"swin_b_all_new"),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# Save best model + processor
trainer.model.save_pretrained(str(project_root / "models" / f"swin_b_finetuned_new"))
