# RainBot

## Project Structure

```
rainbot/
├── src/                    # Source code for training and utilities
├── datasets/              # Dataset storage and processing
├── models/                # Trained model checkpoints
├── notebooks/            # Jupyter notebooks for analysis and experiments
├── pre_processing/       # Data preprocessing scripts
├── data_extraction/      # Data collection and extraction utilities
├── cache/                # Cached data and intermediate results
├── logs/                 # Training logs and metrics
└── plots/                # Generated plots and visualizations
```

## Setup Instructions

### 1. Download Required Data and Models

Before running any training or inference, you need to download the necessary datasets and models. The project uses Wasabi (S3-compatible) storage for hosting these files.

To download the required files, run one of the following scripts:

```bash
# To download models
python models/download_wasabi_models.py

# To download datasets
python datasets/download_wasabi_data.py
```

These scripts will download:
- Pre-trained models for different segments (road, vegetation, terrain)
- Training, validation, and test datasets
- Segmented datasets for specialized training

### 2. Project Components

- **Models**: The project supports multiple model architectures:
  - ResNet50 for country classification
  - Swin Transformer for improved accuracy
  - Ensemble models combining multiple architectures
  - Specialized models for different image segments

- **Datasets**: The project works with:
  - Country-level classification (70 countries)
  - Continent-level classification (6 continents)
  - Segmented datasets (road, vegetation, terrain)

- **Training Scripts**: Located in `src/`:
  - `train.py`: Main training script for country classification
  - `train_segments.py`: Training for segmented data
  - `train_vit.py`: Vision Transformer training
  - `train_segments_vit.py`: Vision Transformer training for segments

### 3. Development

The project includes Jupyter notebooks in the `notebooks/` directory for:
- Model analysis and evaluation
- Experimentation with different architectures
- Data visualization and analysis
- Ensemble model development

## Notes

- The project uses PyTorch and HuggingFace Transformers for model development
- Generated plots and visualizations are stored in `plots/`
- Cache directory is used for storing intermediate results to speed up processing

