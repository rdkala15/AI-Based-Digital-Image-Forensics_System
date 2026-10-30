"""
Evaluate the trained AI-Based Digital Image Forensics model.

Usage:
    python evaluate.py --checkpoint checkpoints\best_model.pt
"""

import argparse
import os

import torch
from PIL import Image

from model import AIGIForensicModel, CLASS_NAMES
from modules.preprocessing import ForensicPreprocessor, PreprocessConfig


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AI image forensic model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="samples"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def collect_images(data_dir):

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "ai-generated")

    if not os.path.exists(real_dir):
        raise FileNotFoundError(
            f"Real image folder not found: {real_dir}"
        )

    if not os.path.exists(fake_dir):
        raise FileNotFoundError(
            f"AI-generated image folder not found: {fake_dir}"
        )

    images = []

    # Real = label 0
    for filename in sorted(os.listdir(real_dir)):

        path = os.path.join(real_dir, filename)

        if os.path.isfile(path):
            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:
                images.append((path, 0))

    # Fake = label 1
    for filename in sorted(os.listdir(fake_dir)):

        path = os.path.join(fake_dir, filename)

        if os.path.isfile(path):
            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:
                images.append((path, 1))

    return images


def main():

    args = parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("AI-BASED DIGITAL IMAGE FORENSICS")
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data       : {args.data_dir}")

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    checkpoint_args = checkpoint.get("args", {})

    backbone_name = checkpoint_args.get(
        "backbone",
        "convnext_tiny"
    )

    image_size = checkpoint_args.get(
        "image_size",
        128
    )

    freeze_backbone = checkpoint_args.get(
        "freeze_backbone",
        False
    )

    print(f"Backbone   : {backbone_name}")
    print(f"Image size : {image_size}")

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = AIGIForensicModel(
        backbone_name=backbone_name,
        pretrained=False,
        freeze_backbone=freeze_backbone
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # Preprocessor
    # --------------------------------------------------

    preprocess_config = PreprocessConfig(
        image_size=image_size,
        freq_size=image_size
    )

    preprocessor = ForensicPreprocessor(
        preprocess_config
    )

    # --------------------------------------------------
    # Collect images
    # --------------------------------------------------

    images = collect_images(args.data_dir)

    real_count = sum(1 for _, label in images if label == 0)
    fake_count = sum(1 for _, label in images if label == 1)

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(f"Real images : {real_count}")
    print(f"Fake images : {fake_count}")
    print(f"Total       : {len(images)}")

    if len(images) == 0:
        print("No images found.")
        return

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    y_true = []
    y_pred = []

    real_correct = 0
    real_total = 0

    fake_correct = 0
    fake_total = 0

    print()
    print("=" * 70)
    print("INDIVIDUAL PREDICTIONS")
    print("=" * 70)

    for index, (image_path, true_label) in enumerate(images, start=1):

        try:

            image = Image.open(image_path)

            if image.mode != "RGB":
                image = image.convert("RGB")

            display_img, spatial_tensor, freq_tensor = \
                preprocessor.process(image)

            spatial_tensor = spatial_tensor.unsqueeze(0).to(device)
            freq_tensor = freq_tensor.unsqueeze(0).to(device)

            with torch.no_grad():

                logits, _, _ = model(
                    spatial_tensor,
                    freq_tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                prediction = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

            confidence = (
                probabilities[0][prediction].item() * 100
            )

            y_true.append(true_label)
            y_pred.append(prediction)

            filename = os.path.basename(image_path)

            status = "CORRECT" if prediction == true_label else "WRONG"

            print(
                f"{index:03d}. "
                f"{filename:<25} "
                f"Actual: {CLASS_NAMES[true_label]:<25} "
                f"Predicted: {CLASS_NAMES[prediction]:<25} "
                f"{confidence:6.2f}%  "
                f"[{status}]"
            )

            if true_label == 0:
                real_total += 1

                if prediction == 0:
                    real_correct += 1

            else:
                fake_total += 1

                if prediction == 1:
                    fake_correct += 1

        except Exception as e:

            print(
                f"ERROR processing {image_path}: {e}"
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total = len(y_true)

    correct = sum(
        1 for true, pred in zip(y_true, y_pred)
        if true == pred
    )

    accuracy = correct / total if total else 0.0

    # Confusion matrix
    #
    #                 Predicted
    #                 Real   Fake
    #
    # Actual Real     TN     FP
    # Actual Fake     FN     TP

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for true, pred in zip(y_true, y_pred):

        if true == 0 and pred == 0:
            tn += 1

        elif true == 0 and pred == 1:
            fp += 1

        elif true == 1 and pred == 0:
            fn += 1

        elif true == 1 and pred == 1:
            tp += 1

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    real_accuracy = (
        real_correct / real_total
        if real_total > 0
        else 0.0
    )

    fake_accuracy = (
        fake_correct / fake_total
        if fake_total > 0
        else 0.0
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(f"Total Images       : {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect           : {total - correct}")

    print()
    print(f"Overall Accuracy   : {accuracy * 100:.2f}%")
    print(f"Precision          : {precision * 100:.2f}%")
    print(f"Recall             : {recall * 100:.2f}%")
    print(f"F1-Score           : {f1 * 100:.2f}%")

    print()
    print(f"Real Accuracy      : {real_accuracy * 100:.2f}%")
    print(f"Fake Accuracy      : {fake_accuracy * 100:.2f}%")

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print()
    print("                 Predicted")
    print("                 Real    Fake")
    print()
    print(f"Actual Real      {tn:4d}    {fp:4d}")
    print(f"Actual Fake      {fn:4d}    {tp:4d}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        f"True Negatives  : {tn} "
        "(Real correctly identified)"
    )

    print(
        f"False Positives : {fp} "
        "(Real incorrectly classified as Fake)"
    )

    print(
        f"False Negatives : {fn} "
        "(Fake incorrectly classified as Real)"
    )

    print(
        f"True Positives   : {tp} "
        "(Fake correctly identified)"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()