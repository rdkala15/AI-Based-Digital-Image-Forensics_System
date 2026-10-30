"""
Module 6: Final AI-Based Digital Image Forensics Inference

This script:
1. Loads the trained best model
2. Accepts an input image
3. Preprocesses the image
4. Performs spatial and frequency analysis
5. Predicts Real or AI-Generated/Manipulated
6. Displays confidence scores
7. Generates a forensic report

Usage:
    python infer.py --image samples/ai-generated/ai1.png

Custom checkpoint:
    python infer.py --image samples/ai-generated/ai1.png ^
        --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from PIL import Image

from model import AIGIForensicModel, CLASS_NAMES
from modules.preprocessing import (
    ForensicPreprocessor,
    PreprocessConfig
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):
    """
    Load the trained forensic model from a checkpoint.
    """

    print("\nLoading trained model...")

    checkpoint = torch.load(
        checkpoint_path,
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

    image_size = checkpoint_args.get(
        "image_size",
        224
    )

    print(f"Backbone   : {backbone_name}")
    print(f"Image size : {image_size}")

    # Create model architecture
    model = AIGIForensicModel(
        backbone_name=backbone_name,
        pretrained=False,
        freeze_backbone=False
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return model, image_size, checkpoint


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(
    model,
    image_path,
    image_size,
    device
):
    """
    Perform forensic analysis on one image.
    """

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_width, original_height = image.size

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    config = PreprocessConfig(
        image_size=image_size,
        freq_size=image_size,
        augment=False
    )

    preprocessor = ForensicPreprocessor(
        config
    )

    _, spatial_tensor, freq_tensor = (
        preprocessor.process(image)
    )

    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits, fused_embedding, gate_weights = model(
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

    # --------------------------------------------------------
    # EXTRACT PROBABILITIES
    # --------------------------------------------------------

    real_probability = (
        probabilities[0, 0].item() * 100
    )

    fake_probability = (
        probabilities[0, 1].item() * 100
    )

    confidence = (
        probabilities[0, prediction].item() * 100
    )

    # --------------------------------------------------------
    # EXTRACT GATE WEIGHTS
    # --------------------------------------------------------

    gates = (
        gate_weights
        .detach()
        .cpu()
        .numpy()[0]
    )

    return {

        "prediction": prediction,

        "prediction_name":
            CLASS_NAMES[prediction],

        "confidence": confidence,

        "real_probability":
            real_probability,

        "fake_probability":
            fake_probability,

        "gate_weights":
            gates,

        "image_size":
            image_size,

        "original_width":
            original_width,

        "original_height":
            original_height,

        "fused_embedding_size":
            fused_embedding.shape[-1]
    }


# ============================================================
# CREATE FORENSIC REPORT
# ============================================================

def create_report(
    image_path,
    result,
    checkpoint,
    output_dir
):
    """
    Generate a text-based forensic analysis report.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    report_path = os.path.join(
        output_dir,
        f"{base_name}_forensic_report.txt"
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    val_accuracy = checkpoint.get(
        "val_acc",
        None
    )

    epoch = checkpoint.get(
        "epoch",
        None
    )

    checkpoint_args = checkpoint.get(
        "args",
        {}
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        # ====================================================
        # HEADER
        # ====================================================

        f.write("=" * 70 + "\n")
        f.write("AI-BASED DIGITAL IMAGE FORENSICS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # ====================================================
        # ANALYSIS INFORMATION
        # ====================================================

        f.write("-" * 70 + "\n")
        f.write("ANALYSIS INFORMATION\n")
        f.write("-" * 70 + "\n\n")

        f.write(
            f"Analysis Date : {timestamp}\n"
        )

        f.write(
            f"Input Image   : {image_path}\n"
        )

        f.write(
            f"Original Size : "
            f"{result['original_width']} x "
            f"{result['original_height']}\n"
        )

        f.write(
            f"Model Input   : "
            f"{result['image_size']} x "
            f"{result['image_size']}\n\n"
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        f.write("-" * 70 + "\n")
        f.write("FORENSIC PREDICTION\n")
        f.write("-" * 70 + "\n\n")

        f.write(
            f"Classification : "
            f"{result['prediction_name']}\n"
        )

        f.write(
            f"Confidence     : "
            f"{result['confidence']:.2f}%\n\n"
        )

        f.write(
            f"Real Probability                : "
            f"{result['real_probability']:.2f}%\n"
        )

        f.write(
            f"AI/Manipulated Probability      : "
            f"{result['fake_probability']:.2f}%\n\n"
        )

        # ====================================================
        # FUSION INFORMATION
        # ====================================================

        f.write("-" * 70 + "\n")
        f.write("SPATIAL-FREQUENCY FEATURE FUSION\n")
        f.write("-" * 70 + "\n\n")

        f.write(
            "The model analyzes the image using two complementary "
            "feature domains:\n\n"
        )

        f.write(
            "1. Spatial Features:\n"
        )

        f.write(
            "   Visual structures, textures, edges, and image patterns.\n\n"
        )

        f.write(
            "2. Frequency Features:\n"
        )

        f.write(
            "   Frequency-domain patterns that may reveal "
            "AI-generation or manipulation artifacts.\n\n"
        )

        f.write(
            "The extracted features are combined using the "
            "fusion classification module.\n\n"
        )

        f.write(
            "Fusion Gate Weights:\n"
        )

        for i, gate in enumerate(
            result["gate_weights"]
        ):

            f.write(
                f"Gate {i + 1} : "
                f"{gate:.4f}\n"
            )

        f.write("\n")

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        f.write("-" * 70 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 70 + "\n\n")

        f.write(
            f"Backbone       : "
            f"{checkpoint_args.get('backbone', 'Unknown')}\n"
        )

        f.write(
            f"Input Size     : "
            f"{result['image_size']} x "
            f"{result['image_size']}\n"
        )

        f.write(
            f"Number Classes : 2\n"
        )

        f.write(
            f"Fused Feature Size : "
            f"{result['fused_embedding_size']}\n"
        )

        if epoch is not None:

            f.write(
                f"Best Epoch     : "
                f"{epoch}\n"
            )

        if val_accuracy is not None:

            f.write(
                f"Best Validation Accuracy : "
                f"{val_accuracy * 100:.2f}%\n"
            )

        f.write("\n")

        # ====================================================
        # INTERPRETATION
        # ====================================================

        f.write("-" * 70 + "\n")
        f.write("FORENSIC INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        if result["prediction"] == 0:

            f.write(
                "RESULT: REAL IMAGE\n\n"
            )

            f.write(
                "The trained model found stronger evidence "
                "consistent with characteristics learned from "
                "real images than with AI-generated image artifacts.\n"
            )

        else:

            f.write(
                "RESULT: AI-GENERATED / MANIPULATED IMAGE\n\n"
            )

            f.write(
                "The trained model found stronger evidence "
                "consistent with patterns learned from "
                "AI-generated or manipulated images.\n"
            )

        f.write("\n\n")

        # ====================================================
        # DISCLAIMER
        # ====================================================

        f.write("-" * 70 + "\n")
        f.write("IMPORTANT NOTE\n")
        f.write("-" * 70 + "\n\n")

        f.write(
            "This prediction is generated using a machine learning "
            "based digital image forensic system. The result should "
            "be treated as forensic assistance and not as absolute "
            "proof of image authenticity or manipulation.\n"
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="AI-Based Digital Image Forensics Inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image for forensic analysis"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reports",
        help="Directory to save forensic reports"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="Device to use: cpu or cuda"
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        args.device
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()

    print("=" * 70)
    print("AI-BASED DIGITAL IMAGE FORENSICS")
    print("MODULE 6: FINAL FORENSIC ANALYSIS")
    print("=" * 70)

    print(
        f"Device     : {device}"
    )

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    print(
        f"Image      : {args.image}"
    )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    if not os.path.exists(
        args.checkpoint
    ):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{args.checkpoint}"
        )

    if not os.path.exists(
        args.image
    ):

        raise FileNotFoundError(
            f"\nImage not found:\n"
            f"{args.image}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, image_size, checkpoint = load_model(
        args.checkpoint,
        device
    )

    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    print("\nAnalyzing image...")

    result = analyze_image(
        model,
        args.image,
        image_size,
        device
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()

    print("=" * 70)
    print("FORENSIC ANALYSIS RESULT")
    print("=" * 70)

    print()

    print(
        f"Classification : "
        f"{result['prediction_name']}"
    )

    print(
        f"Confidence     : "
        f"{result['confidence']:.2f}%"
    )

    print()

    print(
        f"Real Probability : "
        f"{result['real_probability']:.2f}%"
    )

    print(
        f"AI Probability   : "
        f"{result['fake_probability']:.2f}%"
    )

    # ========================================================
    # FUSION INFORMATION
    # ========================================================

    print()

    print("=" * 70)
    print("FEATURE FUSION INFORMATION")
    print("=" * 70)

    print()

    print(
        "Fusion Gate Weights:"
    )

    for i, gate in enumerate(
        result["gate_weights"]
    ):

        print(
            f"Gate {i + 1}: "
            f"{gate:.4f}"
        )

    # ========================================================
    # CREATE REPORT
    # ========================================================

    report_path = create_report(
        args.image,
        result,
        checkpoint,
        args.output_dir
    )

    # ========================================================
    # REPORT OUTPUT
    # ========================================================

    print()

    print("=" * 70)
    print("FORENSIC REPORT GENERATED")
    print("=" * 70)

    print()

    print(
        f"Report saved to:\n"
        f"{report_path}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()