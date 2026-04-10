# Plant Seedlings Classification using CNN

A Convolutional Neural Network for classifying 12 species of plant seedlings, built with TensorFlow/Keras.

## Overview

This project uses a custom CNN architecture to classify plant seedling images into 12 species. The model uses data augmentation, batch normalization, and progressive dropout to achieve competitive accuracy on the [Plant Seedlings Dataset](https://www.kaggle.com/c/plant-seedlings-classification).

## Species

The model classifies the following 12 plant species:

| | | |
|---|---|---|
| Black-grass | Charlock | Cleavers |
| Common Chickweed | Common wheat | Fat Hen |
| Loose Silky-bent | Maize | Scentless Mayweed |
| Shepherds Purse | Small-flowered Cranesbill | Sugar beet |

## Model Architecture

```
Conv2D(32, 5×5) → BatchNorm → MaxPool → Dropout(0.2)
Conv2D(64, 5×5) → BatchNorm → MaxPool → Dropout(0.3)
Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout(0.4)
Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout(0.5)
GlobalMaxPooling2D → Dense(256) → Dropout(0.5) → Dense(12, softmax)
```

**Training features:**
- Data augmentation (rotation, shift, flip, zoom)
- Adam optimizer with ReduceLROnPlateau
- Early stopping with best weight restoration

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is from the [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) Kaggle competition, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Download from Kaggle and extract into a `data/` directory:

```bash
# Option 1: Using Kaggle CLI
pip install kaggle
kaggle competitions download -c plant-seedlings-classification
unzip plant-seedlings-classification.zip -d data/

# Option 2: Download manually from
# https://www.kaggle.com/c/plant-seedlings-classification/data
# and extract into data/
```

Expected folder structure:

```
data/
├── train/
│   ├── Black-grass/        (263 images)
│   ├── Charlock/           (390 images)
│   ├── Cleavers/           (287 images)
│   ├── Common Chickweed/   (611 images)
│   ├── Common wheat/       (221 images)
│   ├── Fat Hen/            (475 images)
│   ├── Loose Silky-bent/   (654 images)
│   ├── Maize/              (221 images)
│   ├── Scentless Mayweed/  (516 images)
│   ├── Shepherds Purse/    (231 images)
│   ├── Small-flowered Cranesbill/ (496 images)
│   └── Sugar beet/         (385 images)
├── test/                   (794 unlabeled images)
└── sample_submission.csv
```

Total: **4,750 labeled training images** across 12 classes.

## Usage

### Training

Run the Jupyter notebook:

```bash
jupyter notebook "computer vision.ipynb"
```

The notebook loads raw PNG images directly from the folder structure — no preprocessing scripts needed.

### Demo

After training, launch the Gradio web demo:

```bash
python app.py
```

This starts a local web app where you can upload a plant seedling image and get the predicted species.

## Project Structure

```
├── computer vision.ipynb   # Training notebook
├── app.py                  # Gradio demo application
├── Labels.csv              # Dataset labels reference
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── data/                   # Dataset (not tracked in git)
│   ├── train/              #   Labeled training images by species
│   ├── test/               #   Unlabeled test images
│   └── sample_submission.csv
└── model/                  # Saved model (generated after training)
    ├── plant_seedling_model.keras
    └── class_names.npy
```

## License

This project is for educational purposes.

Dataset: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — provided by the [Plant Seedlings Dataset](https://www.kaggle.com/c/plant-seedlings-classification) Kaggle competition.
