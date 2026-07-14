"""
Sign Language Alphabet Detection - Training Script
===================================================
Trains the MobileNetV2 model on a folder-structured dataset.

Expected dataset layout:
    dataset/
    ├── A/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── B/
    │   ├── img001.jpg
    │   └── ...
    └── Z/
        └── ...

Usage:
    python train.py --dataset ./dataset --epochs 20 --batch_size 32
"""

import os
import argparse
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt

from model import build_model, IMG_SIZE, NUM_CLASSES, LABELS


# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train Sign Language Detector')
    p.add_argument('--dataset',    type=str,   required=True,
                   help='Path to dataset root (with A-Z subfolders)')
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--val_split',  type=float, default=0.2,
                   help='Fraction of data used for validation')
    p.add_argument('--fine_tune',  action='store_true',
                   help='Unfreeze last 30 backbone layers for fine-tuning')
    p.add_argument('--output_dir', type=str,   default='./output',
                   help='Directory to save model & training artifacts')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  Data Pipeline
# ══════════════════════════════════════════════════════════════════════

def create_data_generators(dataset_path: str,
                           batch_size: int,
                           val_split: float):
    """
    Create training and validation data generators with augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0,                     # raw pixels; preprocessing in model
        validation_split=val_split,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=False,           # sign language is NOT horizontally symmetric
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen


# ══════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════

def train(args):
    print("=" * 60)
    print("  Sign Language Alphabet Detector — Training")
    print("=" * 60)

    # ── Output directory ──────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # ── Data generators ───────────────────────────────────────────────
    print(f"\n[INFO] Loading dataset from: {args.dataset}")
    train_gen, val_gen = create_data_generators(
        args.dataset, args.batch_size, args.val_split
    )

    num_train = train_gen.samples
    num_val   = val_gen.samples
    print(f"   Training samples:   {num_train}")
    print(f"   Validation samples: {num_val}")
    print(f"   Classes found:      {train_gen.num_classes}")

    # Verify class count
    if train_gen.num_classes != NUM_CLASSES:
        print(f"[WARNING] Expected {NUM_CLASSES} classes, found {train_gen.num_classes}")

    # Save class-index mapping
    class_map_path = os.path.join(run_dir, 'class_indices.json')
    with open(class_map_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"   Class mapping saved to: {class_map_path}")

    # ── Build model ───────────────────────────────────────────────────
    fine_tune_from = -30 if args.fine_tune else None   # last 30 layers
    model = build_model(
        num_classes=train_gen.num_classes,
        fine_tune_from=fine_tune_from
    )

    # Update learning rate if specified
    if args.lr != 1e-3:
        model.optimizer.learning_rate.assign(args.lr)

    print(f"\n[MODEL] {model.name}")
    print(f"   Input shape: {(IMG_SIZE, IMG_SIZE, 3)}")
    print(f"   Fine-tuning: {'Yes (last 30 layers)' if args.fine_tune else 'No (frozen backbone)'}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(run_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(run_dir, 'logs')
        )
    ]

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n[START] Starting training for {args.epochs} epochs...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(run_dir, 'final_model.keras')
    model.save(final_path)
    print(f"\n[SAVED] Final model saved to: {final_path}")

    # Also save as TFLite for lightweight deployment
    save_tflite(model, run_dir)

    # ── Plot training curves ──────────────────────────────────────────
    plot_history(history, run_dir)

    print(f"\n[DONE] Training complete! All artifacts saved to: {run_dir}")
    return model, history


# ══════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════

def save_tflite(model, output_dir):
    """Convert and save a TFLite version of the model."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = os.path.join(output_dir, 'sign_language_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"[SAVED] TFLite model saved to: {tflite_path}")
    except Exception as e:
        print(f"[WARNING] TFLite conversion failed: {e}")


def plot_history(history, output_dir):
    """Save accuracy & loss plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'],     label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'],  label='Val Accuracy',   linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'],     label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'],  label='Val Loss',   linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[SAVED] Training curves saved to: {plot_path}")


# ══════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    args = parse_args()
    train(args)
