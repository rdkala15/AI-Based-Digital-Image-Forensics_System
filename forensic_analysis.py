"""
FINAL END-TO-END DIGITAL IMAGE FORENSIC ANALYSIS
------------------------------------------------

Pipeline:

    Input Image
         |
         v
    Preprocessing
         |
         +----------------------+
         |                      |
         v                      v
    ConvNeXt-Tiny          FFT + Haar DWT
    Spatial Features       Frequency Features
         |                      |
         +----------+-----------+
                    |
                    v
              Feature Fusion
                    |
                    v
             Binary Classifier
                    |
                    v
             Real / AI-Generated
                    |
                    v
                 Grad-CAM
                    |
                    v
              Forensic PDF

Usage:

    python forensic_analysis.py --image dataset/ai-generated/ai1.png

Optional:

    python forensic_analysis.py \
        --image dataset/real/real1.jpg \
        --checkpoint checkpoints/best_model.pt
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

from predict import (
    GradCAM,
    save_gradcam
)

from report import generate_report


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Complete AI-Based Digital Image "
            "Forensic Analysis"
        )
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
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="cuda or cpu"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for Grad-CAM and heatmap"
    )

    parser.add_argument(
        "--report_dir",
        type=str,
        default="reports",
        help="Directory for forensic PDF"
    )

    return parser.parse_args()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if not os.path.exists(args.image):

        raise FileNotFoundError(
            f"Image not found:\n{args.image}"
        )

    if not os.path.exists(args.checkpoint):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{args.checkpoint}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    os.makedirs(
        args.report_dir,
        exist_ok=True
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        args.device
    )

    print()
    print("=" * 70)
    print(
        "AI-BASED DIGITAL IMAGE FORENSIC ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Device     : {device}"
    )

    print(
        f"Image      : {args.image}"
    )

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print(
        "[1/7] Loading trained model..."
    )

    checkpoint = torch.load(
        args.checkpoint,
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

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = AIGIForensicModel(

        backbone_name=backbone_name,

        pretrained=False,

        freeze_backbone=freeze_backbone

    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "      Model loaded successfully."
    )

    print(
        f"      Backbone   : {backbone_name}"
    )

    print(
        f"      Image size : {image_size}"
    )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    print()
    print(
        "[2/7] Loading input image..."
    )

    image = Image.open(
        args.image
    ).convert("RGB")

    print(
        f"      Original size: {image.size}"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    print()
    print(
        "[3/7] Preprocessing image..."
    )

    preprocess_config = PreprocessConfig(

        image_size=image_size,

        freq_size=image_size

    )

    preprocessor = ForensicPreprocessor(
        preprocess_config
    )

    (
        display_img,
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

    print(
        f"      Spatial tensor   : "
        f"{spatial_tensor.shape}"
    )

    print(
        f"      Frequency tensor : "
        f"{freq_tensor.shape}"
    )

    # ========================================================
    # GRAD-CAM + CLASSIFICATION
    # ========================================================

    print()
    print(
        "[4/7] Running forensic classification..."
    )

    gradcam = GradCAM(
        model
    )

    try:

        (
            cam,
            logits,
            probabilities,
            gates
        ) = gradcam.generate(

            spatial_tensor,

            freq_tensor

        )

    finally:

        gradcam.close()

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[
            0,
            prediction
        ].item()
        * 100.0
    )

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

    if 0 <= prediction < len(CLASS_NAMES):
        prediction_name = CLASS_NAMES[prediction]
    else:
        prediction_name = f"Class {prediction}"

    print(
        "      Classification completed."
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "FORENSIC CLASSIFICATION RESULT"
    )
    print(
        "=" * 70
    )

    print(
        f"Prediction       : {prediction_name}"
    )

    print(
        f"Confidence       : {confidence:.2f}%"
    )

    print(
        f"Real Probability : {real_probability:.2f}%"
    )

    print(
        f"Fake Probability : {fake_probability:.2f}%"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    print()
    print(
        "[5/7] Generating artifact localization..."
    )

    image_name = os.path.splitext(
        os.path.basename(
            args.image
        )
    )[0]

    gradcam_path = os.path.join(

        args.output_dir,

        f"{image_name}_gradcam.png"

    )

    (
        saved_gradcam_path,
        saved_heatmap_path
    ) = save_gradcam(

        image,

        cam,

        gradcam_path

    )

    print(
        f"      Grad-CAM : {saved_gradcam_path}"
    )

    print(
        f"      Heatmap  : {saved_heatmap_path}"
    )

    # ========================================================
    # PDF REPORT
    # ========================================================

    print()
    print(
        "[6/7] Generating forensic PDF report..."
    )

    report_path = os.path.join(

        args.report_dir,

        f"{image_name}_forensic_report.pdf"

    )

    generate_report(

        image_path=args.image,

        checkpoint_path=args.checkpoint,

        output_path=report_path,

        prediction=prediction_name,

        confidence=confidence,

        real_probability=real_probability,

        fake_probability=fake_probability,

        gradcam_path=saved_gradcam_path,

        heatmap_path=saved_heatmap_path

    )

    print(
        f"      PDF : {report_path}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "[7/7] Analysis completed."
    )

    print()
    print("=" * 70)
    print(
        "FORENSIC ANALYSIS COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "FINAL RESULT"
    )

    print(
        f"  Classification : {prediction_name}"
    )

    print(
        f"  Confidence     : {confidence:.2f}%"
    )

    print(
        f"  Real           : {real_probability:.2f}%"
    )

    print(
        f"  AI/Fake        : {fake_probability:.2f}%"
    )

    print()
    print(
        "GENERATED FILES"
    )

    print(
        f"  Grad-CAM : {saved_gradcam_path}"
    )

    print(
        f"  Heatmap  : {saved_heatmap_path}"
    )

    print(
        f"  PDF      : {report_path}"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()