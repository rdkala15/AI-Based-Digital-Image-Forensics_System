"""
Module 5: Grad-CAM Visualization

Usage:

python gradcam_visualize.py ^
    --image samples/ai-generated/ai1.png
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

from model import AIGIForensicModel, CLASS_NAMES

from modules.preprocessing import (
    ForensicPreprocessor,
    PreprocessConfig
)

from modules.gradcam import (
    GradCAMExplainer,
    resize_heatmap,
    overlay_heatmap,
    create_heatmap_image
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    device
):

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

    model = AIGIForensicModel(

        backbone_name=backbone_name,

        pretrained=False,

        freeze_backbone=False

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return (
        model,
        image_size
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Module 5: Grad-CAM Visualization"
    )

    parser.add_argument(

        "--image",

        type=str,

        required=True,

        help="Path to image"

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

        default="outputs/gradcam",

        help="Directory for Grad-CAM results"

    )

    parser.add_argument(

        "--device",

        type=str,

        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "AI-BASED DIGITAL IMAGE FORENSICS"
    )

    print(
        "MODULE 5: GRAD-CAM EXPLAINABILITY"
    )

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

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(
        args.image
    ):

        raise FileNotFoundError(

            f"Image not found:\n"
            f"{args.image}"

        )

    if not os.path.exists(
        args.checkpoint
    ):

        raise FileNotFoundError(

            f"Checkpoint not found:\n"
            f"{args.checkpoint}"

        )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(

        args.output_dir,

        exist_ok=True

    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading trained model..."
    )

    model, image_size = load_model(

        args.checkpoint,

        device

    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    image = Image.open(
        args.image
    ).convert(
        "RGB"
    )

    original_width, original_height = (
        image.size
    )

    # --------------------------------------------------------
    # PREPROCESS IMAGE
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
        preprocessor.process(
            image
        )
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

    # --------------------------------------------------------
    # CREATE GRAD-CAM
    # --------------------------------------------------------

    print()

    print(
        "Generating Grad-CAM visualization..."
    )

    gradcam = GradCAMExplainer(
        model
    )

    heatmap, prediction, probabilities = (
        gradcam.generate(

            spatial_tensor,

            freq_tensor

        )
    )

    # Remove hooks

    gradcam.remove_hooks()

    # --------------------------------------------------------
    # RESIZE HEATMAP
    # --------------------------------------------------------

    resized_heatmap = resize_heatmap(

        heatmap,

        original_width,

        original_height

    )

    # --------------------------------------------------------
    # CREATE HEATMAP IMAGE
    # --------------------------------------------------------

    heatmap_image = create_heatmap_image(
        resized_heatmap
    )

    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    overlay_image = overlay_heatmap(

        image,

        resized_heatmap,

        alpha=0.45

    )

    # --------------------------------------------------------
    # CREATE FILENAMES
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(
            args.image
        )
    )[0]

    heatmap_path = os.path.join(

        args.output_dir,

        f"{base_name}_gradcam_heatmap.png"

    )

    overlay_path = os.path.join(

        args.output_dir,

        f"{base_name}_gradcam_overlay.png"

    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    heatmap_image.save(
        heatmap_path
    )

    overlay_image.save(
        overlay_path
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "GRAD-CAM RESULT"
    )

    print("=" * 70)

    print()

    print(

        f"Prediction : "
        f"{CLASS_NAMES[prediction]}"

    )

    print(

        f"Confidence : "
        f"{probabilities[prediction] * 100:.2f}%"

    )

    print()

    print(

        f"Real Probability : "
        f"{probabilities[0] * 100:.2f}%"

    )

    print(

        f"AI Probability   : "
        f"{probabilities[1] * 100:.2f}%"

    )

    print()

    print("=" * 70)

    print(
        "GRAD-CAM FILES GENERATED"
    )

    print("=" * 70)

    print()

    print(
        f"Heatmap:\n"
        f"{heatmap_path}"
    )

    print()

    print(
        f"Overlay:\n"
        f"{overlay_path}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()