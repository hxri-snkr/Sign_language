# Sign Language Alphabet Detector

Real-time ASL alphabet (A–Z) detection using MobileNetV2 + TensorFlow, designed for children's learning.

## Project Structure

```
sign-language-detector/
├── model.py              # MobileNetV2 model architecture
├── train.py              # Training pipeline
├── predict.py            # Real-time webcam inference
├── prepare_dataset.py    # Video → image frame extractor
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

If you have raw video files (one per letter):

```bash
python prepare_dataset.py --input ./raw_videos --output ./dataset --fps 12
```

Your dataset should end up looking like:
```
dataset/
├── A/   (images of sign "A")
├── B/   (images of sign "B")
├── ...
└── Z/   (images of sign "Z")
```

### 3. Train the Model

```bash
# Basic training (frozen backbone)
python train.py --dataset ./dataset --epochs 20 --batch_size 32

# With fine-tuning (better accuracy, slower)
python train.py --dataset ./dataset --epochs 30 --batch_size 16 --fine_tune
```

### 4. Run Real-time Detection

```bash
# Basic webcam detection
python predict.py --model ./output/run_XXXXX/best_model.keras

# With text-to-speech
python predict.py --model ./output/run_XXXXX/best_model.keras --speak

# Adjust confidence threshold
python predict.py --model ./output/run_XXXXX/best_model.keras --confidence 0.7
```

## Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (26, Softmax)  →  A–Z prediction
```

## Webcam Controls

| Key | Action |
|-----|--------|
| `Q` | Quit   |

## Output Files

After training, the `output/run_XXXXX/` folder contains:
- `best_model.keras` — Best model checkpoint
- `final_model.keras` — Final epoch model
- `sign_language_model.tflite` — Lightweight mobile model
- `class_indices.json` — Class label mapping
- `training_curves.png` — Accuracy/loss plots
- `logs/` — TensorBoard logs
