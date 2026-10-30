"""
Independent / External Real-World Dataset Evaluation
------------------------------------------------------

Evaluates the trained AI-Based Digital Image Forensics model
on a completely separate external dataset stored in:

    external_test/
        ai-generated/
        real/

This does NOT retrain the model.

It calculates:

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    - Per-image prediction results

Results are saved separately to:

    external_realtime_evaluation_results.csv

Usage:

    python external_realtime_evaluation.py
"""

from __future__ import annotations

import csv
import os

import torch
from PIL import Image

from model import AIGIForensicModel, CLASS_NAMES
from modules.preprocessing import (
    ForensicPreprocessor,
    PreprocessConfig
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Use the separate external dataset.
# Do NOT change this back to "dataset".
DATASET_DIR = "external_test"

AI_DIR = os.path.join(
    DATASET_DIR,
    "ai-generated"
)

REAL_DIR = os.path.join(
    DATASET_DIR,
    "real"
)

# Latest trained model
CHECKPOINT_PATH = (
    "checkpoints/best_model.pt"
)

# Separate CSV so the previous results are preserved
RESULTS_CSV = (
    "external_realtime_evaluation_results.csv"
)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"

}


# ============================================================
# LOAD IMAGE PATHS
# ============================================================

def collect_images(
    directory,
    label
):

    images = []

    if not os.path.exists(directory):

        print(
            f"WARNING: Directory not found: {directory}"
        )

        return images

    for filename in sorted(
        os.listdir(directory)
    ):

        path = os.path.join(
            directory,
            filename
        )

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        images.append(
            (
                path,
                label
            )
        )

    return images


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    total = len(actual)

    correct = sum(
        a == p
        for a, p in zip(
            actual,
            predicted
        )
    )

    accuracy = (

        correct / total * 100

        if total > 0

        else 0.0

    )

    # --------------------------------------------------------
    # Binary classification
    #
    # Class 0 = Real
    # Class 1 = AI-generated
    # --------------------------------------------------------

    true_positive = sum(

        1

        for a, p in zip(
            actual,
            predicted
        )

        if a == 1 and p == 1

    )

    false_positive = sum(

        1

        for a, p in zip(
            actual,
            predicted
        )

        if a == 0 and p == 1

    )

    false_negative = sum(

        1

        for a, p in zip(
            actual,
            predicted
        )

        if a == 1 and p == 0

    )

    true_negative = sum(

        1

        for a, p in zip(
            actual,
            predicted
        )

        if a == 0 and p == 0

    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (

        true_positive
        / (
            true_positive
            + false_positive
        )

        if (
            true_positive
            + false_positive
        ) > 0

        else 0.0

    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (

        true_positive
        / (
            true_positive
            + false_negative
        )

        if (
            true_positive
            + false_negative
        ) > 0

        else 0.0

    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (

        2
        * precision
        * recall
        / (
            precision
            + recall
        )

        if (
            precision
            + recall
        ) > 0

        else 0.0

    )

    return {

        "total": total,

        "correct": correct,

        "incorrect":
            total - correct,

        "accuracy":
            accuracy,

        "precision":
            precision * 100,

        "recall":
            recall * 100,

        "f1":
            f1 * 100,

        "tn":
            true_negative,

        "fp":
            false_positive,

        "fn":
            false_negative,

        "tp":
            true_positive

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 75)

    print(
        "INDEPENDENT / EXTERNAL REAL-WORLD DATASET EVALUATION"
    )

    print("=" * 75)

    # ========================================================
    # DATASET PATHS
    # ========================================================

    print()

    print(
        f"External dataset    : {DATASET_DIR}"
    )

    print(
        f"Real folder         : {REAL_DIR}"
    )

    print(
        f"AI folder           : {AI_DIR}"
    )

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(

            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Please make sure "
            f"checkpoints/best_model.pt "
            f"exists."

        )

    # ========================================================
    # CHECK DATASET DIRECTORIES
    # ========================================================

    if not os.path.exists(
        REAL_DIR
    ):

        raise FileNotFoundError(

            f"\nReal image folder not found:\n"
            f"{REAL_DIR}\n\n"
            f"Create the folder and place "
            f"the real images inside it."

        )

    if not os.path.exists(
        AI_DIR
    ):

        raise FileNotFoundError(

            f"\nAI image folder not found:\n"
            f"{AI_DIR}\n\n"
            f"Create the folder and place "
            f"the AI-generated images inside it."

        )

    # ========================================================
    # COLLECT DATA
    # ========================================================

    real_images = collect_images(

        REAL_DIR,

        0

    )

    ai_images = collect_images(

        AI_DIR,

        1

    )

    all_images = (

        real_images
        + ai_images

    )

    print()

    print(
        f"Real images         : "
        f"{len(real_images)}"
    )

    print(
        f"AI-generated images : "
        f"{len(ai_images)}"
    )

    print(
        f"Total images        : "
        f"{len(all_images)}"
    )

    # ========================================================
    # CHECK DATA BALANCE
    # ========================================================

    if len(real_images) != len(ai_images):

        print()

        print(
            "WARNING:"
        )

        print(
            "The number of real and AI images "
            "is not equal."
        )

        print(
            "This evaluation can still run, "
            "but a balanced dataset is preferred."
        )

    if len(all_images) == 0:

        raise RuntimeError(

            "No supported images found "
            "in external_test."

        )

    # ========================================================
    # LOAD DEVICE
    # ========================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )

    print()

    print(
        f"Device              : {device}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=device

    )

    checkpoint_args = checkpoint.get(

        "args",

        {}

    )

    backbone_name = checkpoint_args.get(

        "backbone",

        "convnext_tiny"

    )

    freeze_backbone = checkpoint_args.get(

        "freeze_backbone",

        False

    )

    image_size = checkpoint_args.get(

        "image_size",

        224

    )

    print()

    print(
        f"Backbone            : "
        f"{backbone_name}"
    )

    print(
        f"Image size          : "
        f"{image_size}"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = AIGIForensicModel(

        backbone_name=backbone_name,

        pretrained=False,

        freeze_backbone=freeze_backbone

    ).to(device)

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )

    model.eval()

    print()

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # PREPROCESSOR
    # ========================================================

    preprocess_config = PreprocessConfig(

        image_size=image_size,

        freq_size=image_size

    )

    preprocessor = ForensicPreprocessor(

        preprocess_config

    )

    # ========================================================
    # EVALUATION STORAGE
    # ========================================================

    actual_labels = []

    predicted_labels = []

    results = []

    # ========================================================
    # EVALUATE
    # ========================================================

    print()

    print(
        "Evaluating external images..."
    )

    print()

    for index, (

        image_path,

        actual_label

    ) in enumerate(

        all_images,

        start=1

    ):

        filename = os.path.basename(

            image_path

        )

        try:

            # ------------------------------------------------
            # LOAD IMAGE
            # ------------------------------------------------

            image = Image.open(

                image_path

            ).convert("RGB")

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            (

                _,

                spatial_tensor,

                freq_tensor

            ) = preprocessor.process(

                image

            )

            spatial_tensor = (

                spatial_tensor

                .unsqueeze(0)

                .to(device)

            )

            freq_tensor = (

                freq_tensor

                .unsqueeze(0)

                .to(device)

            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            with torch.no_grad():

                logits = (

                    model.forensic_classifier(

                        spatial_tensor,

                        freq_tensor

                    )

                )

                probabilities = (

                    torch.softmax(

                        logits,

                        dim=1

                    )

                )

            # ------------------------------------------------
            # PREDICTED CLASS
            # ------------------------------------------------

            predicted_label = (

                torch.argmax(

                    probabilities,

                    dim=1

                ).item()

            )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            real_probability = (

                probabilities[

                    0,

                    0

                ].item()

                * 100.0

            )

            fake_probability = (

                probabilities[

                    0,

                    1

                ].item()

                * 100.0

            )

            confidence = (

                probabilities[

                    0,

                    predicted_label

                ].item()

                * 100.0

            )

            # ------------------------------------------------
            # CLASS NAMES
            # ------------------------------------------------

            actual_name = (

                CLASS_NAMES[

                    actual_label

                ]

            )

            predicted_name = (

                CLASS_NAMES[

                    predicted_label

                ]

            )

            # ------------------------------------------------
            # CORRECT / INCORRECT
            # ------------------------------------------------

            correct = (

                actual_label
                == predicted_label

            )

            # ------------------------------------------------
            # STORE LABELS
            # ------------------------------------------------

            actual_labels.append(

                actual_label

            )

            predicted_labels.append(

                predicted_label

            )

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            results.append({

                "filename":
                    filename,

                "path":
                    image_path,

                "actual":
                    actual_name,

                "predicted":
                    predicted_name,

                "confidence":
                    f"{confidence:.2f}",

                "real_probability":
                    f"{real_probability:.2f}",

                "fake_probability":
                    f"{fake_probability:.2f}",

                "correct":
                    (
                        "YES"
                        if correct
                        else "NO"
                    )

            })

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            status = (

                "[OK]"
                if correct

                else "[X]"

            )

            print(

                f"[{index:02d}/"
                f"{len(all_images):02d}] "
                f"{status} "
                f"{filename:<25} "
                f"Actual: "
                f"{actual_name:<22} "
                f"Predicted: "
                f"{predicted_name:<22} "
                f"Conf: "
                f"{confidence:6.2f}%"

            )

        except Exception as error:

            print()

            print(

                f"ERROR processing "
                f"{image_path}: "
                f"{error}"

            )

    # ========================================================
    # CHECK EVALUATION
    # ========================================================

    if len(actual_labels) == 0:

        raise RuntimeError(

            "No images were successfully evaluated."

        )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    metrics = calculate_metrics(

        actual_labels,

        predicted_labels

    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(

        RESULTS_CSV,

        "w",

        newline="",

        encoding="utf-8"

    ) as csv_file:

        fieldnames = [

            "filename",

            "path",

            "actual",

            "predicted",

            "confidence",

            "real_probability",

            "fake_probability",

            "correct"

        ]

        writer = csv.DictWriter(

            csv_file,

            fieldnames=fieldnames

        )

        writer.writeheader()

        writer.writerows(

            results

        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()

    print("=" * 75)

    print(
        "EXTERNAL / UNSEEN DATASET RESULTS"
    )

    print("=" * 75)

    print()

    print(
        f"Total Images         : "
        f"{metrics['total']}"
    )

    print(
        f"Correct Predictions  : "
        f"{metrics['correct']}"
    )

    print(
        f"Incorrect Predictions: "
        f"{metrics['incorrect']}"
    )

    print()

    print(
        f"Accuracy             : "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"Precision            : "
        f"{metrics['precision']:.2f}%"
    )

    print(
        f"Recall               : "
        f"{metrics['recall']:.2f}%"
    )

    print(
        f"F1 Score             : "
        f"{metrics['f1']:.2f}%"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()

    print(
        "CONFUSION MATRIX"
    )

    print()

    print(
        "                     "
        "Predicted Real   "
        "Predicted AI"
    )

    print(

        f"Actual Real          "
        f"{metrics['tn']:>10}       "
        f"{metrics['fp']:>10}"

    )

    print(

        f"Actual AI            "
        f"{metrics['fn']:>10}       "
        f"{metrics['tp']:>10}"

    )

    print()

    print(
        "TN = Real correctly classified as Real"
    )

    print(
        "FP = Real incorrectly classified as AI"
    )

    print(
        "FN = AI incorrectly classified as Real"
    )

    print(
        "TP = AI correctly classified as AI"
    )

    # ========================================================
    # CLASS-WISE DETECTION
    # ========================================================

    real_total = (

        metrics["tn"]
        + metrics["fp"]

    )

    ai_total = (

        metrics["fn"]
        + metrics["tp"]

    )

    real_detection = (

        metrics["tn"]
        / real_total
        * 100

        if real_total > 0

        else 0.0

    )

    ai_detection = (

        metrics["tp"]
        / ai_total
        * 100

        if ai_total > 0

        else 0.0

    )

    print()

    print(
        "CLASS-WISE DETECTION"
    )

    print()

    print(

        f"Real images correctly detected : "
        f"{metrics['tn']}/{real_total} "
        f"({real_detection:.2f}%)"

    )

    print(

        f"AI images correctly detected   : "
        f"{metrics['tp']}/{ai_total} "
        f"({ai_detection:.2f}%)"

    )

    # ========================================================
    # CSV LOCATION
    # ========================================================

    print()

    print(
        "Detailed per-image results saved to:"
    )

    print(

        RESULTS_CSV

    )

    print()

    print("=" * 75)

    print(
        "INDEPENDENT EXTERNAL EVALUATION COMPLETED"
    )

    print("=" * 75)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()