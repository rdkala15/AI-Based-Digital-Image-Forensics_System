"""
Module 7: Web Interface (Streamlit)

AI-Based Digital Image Forensics System

Forensic-support web interface providing:

1. Evidence upload
2. Evidence identification
3. SHA-256 integrity verification
4. File and metadata analysis
5. Image preprocessing
6. Spatial feature extraction using ConvNeXt-Tiny
7. Frequency-domain analysis using FFT + Haar DWT
8. Spatial-frequency feature fusion
9. Real vs AI-generated/manipulated classification
10. Grad-CAM explanation
11. Suspicious-region visualization
12. Session case history
13. PDF forensic report generation

Important:
This system is an automated forensic assistance/triage tool.
Its findings are not absolute proof of authenticity or manipulation.
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime
import textwrap
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from forensic_integrity import calculate_sha256_bytes
from forensic_metadata import extract_metadata, analyze_metadata
from forensic_manipulation import analyze_manipulation
from copy_move_forensics import detect_copy_move

from forensic_metadata import (
    extract_metadata,
    analyze_metadata,
)

from model import (
    AIGIForensicModel,
    CLASS_NAMES,
)

from modules.preprocessing import (
    ForensicPreprocessor,
    PreprocessConfig,
)

from modules.explainability import (
    GradCAMExplainer,
    ArtifactLocalizer,
)

from modules.report_generator import (
    ForensicCaseResult,
    ForensicReportGenerator,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Best checkpoint from Experiment 4
CHECKPOINT_PATH = "checkpoints/experiment4_best_epoch10.pt"

REPORT_OUTPUT_DIR = "outputs"

SUPPORTED_IMAGE_TYPES = [
    "png",
    "jpg",
    "jpeg",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIGI Forensics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap'
    );

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: "IBM Plex Sans", sans-serif !important;
        font-weight: 700 !important;
    }

    code, .mono {
        font-family: "IBM Plex Mono", monospace !important;
    }

    /* --------------------------------------------------------
       HEADER
       -------------------------------------------------------- */

    .app-header {
        padding: 8px 0 20px 0;
        border-bottom: 1px solid #283136;
        margin-bottom: 24px;
    }

    .app-kicker {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.70rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #62b5d6;
        margin-bottom: 8px;
    }

    .app-title {
        font-size: 2.35rem;
        font-weight: 700;
        line-height: 1.1;
        color: #edf1f2;
        margin-bottom: 8px;
    }

    .app-subtitle {
        color: #8f9da3;
        font-size: 0.98rem;
        max-width: 780px;
        line-height: 1.55;
    }

    .status-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 20px;
        border: 1px solid #303b40;
        background: #151b1e;
        color: #a9b5ba;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem;
    }

    .status-pill strong {
        color: #edf1f2;
    }

    /* --------------------------------------------------------
       SECTION HEADERS
       -------------------------------------------------------- */

    .section-header {
        margin-top: 8px;
        margin-bottom: 14px;
    }

    .section-label {
        font-family: "IBM Plex Mono", monospace;
        color: #62b5d6;
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 3px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #edf1f2;
    }

    .section-description {
        color: #7f8d93;
        font-size: 0.86rem;
        margin-top: 3px;
    }

    /* --------------------------------------------------------
       CASE META
       -------------------------------------------------------- */

    .case-meta {
        background: #101416;
        border: 1px dashed #303b40;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
        display: flex;
        gap: 28px;
        flex-wrap: wrap;
    }

    .case-meta-item {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.72rem;
        color: #8f9da3;
    }

    .case-meta-item strong {
        color: #cdd6d9;
        display: block;
        font-size: 0.82rem;
        margin-top: 2px;
    }

    /* --------------------------------------------------------
       VERDICT
       -------------------------------------------------------- */

    .verdict-card {
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 14px;
        border: 1px solid;
    }

    .verdict-real {
        background: #0e1b17;
        border-color: #2b6d57;
    }

    .verdict-ai {
        background: #21180d;
        border-color: #a16f20;
    }

    .verdict-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.66rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #8f9da3;
        margin-bottom: 5px;
    }

    .verdict-name {
        font-size: 1.65rem;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .verdict-real .verdict-name {
        color: #50c89b;
    }

    .verdict-ai .verdict-name {
        color: #e9a63f;
    }

    .verdict-confidence {
        font-size: 0.82rem;
        color: #9ba8ad;
    }

    /* --------------------------------------------------------
       METRIC CARDS
       -------------------------------------------------------- */

    .metric-card {
        background: #151b1e;
        border: 1px solid #2a3439;
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 86px;
    }

    .metric-value {
        color: #edf1f2;
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-label {
        color: #7f8d93;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* --------------------------------------------------------
       EVIDENCE CARDS
       -------------------------------------------------------- */

    .evidence-card {
        background: #12171a;
        border: 1px solid #2a3439;
        border-radius: 12px;
        padding: 18px;
        margin-top: 16px;
    }

    .evidence-title {
        color: #edf1f2;
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 4px;
    }

    .evidence-subtitle {
        color: #7f8d93;
        font-size: 0.78rem;
        margin-bottom: 12px;
    }

    /* --------------------------------------------------------
       NOTES
       -------------------------------------------------------- */

    .note-box {
        background: #151b1e;
        border: 1px solid #2a3439;
        border-radius: 10px;
        padding: 15px 18px;
        margin-top: 12px;
    }

    .note-item {
        color: #aab5b9;
        font-size: 0.83rem;
        line-height: 1.5;
        margin-bottom: 9px;
    }

    .note-item:last-child {
        margin-bottom: 0;
    }

    .note-item::before {
        content: "▸ ";
        color: #62b5d6;
        font-weight: 600;
    }

    /* --------------------------------------------------------
       DASHBOARD
       -------------------------------------------------------- */

    .dashboard-stat {
        background: #151b1e;
        border: 1px solid #2a3439;
        border-radius: 10px;
        padding: 16px;
    }

    .dashboard-stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #edf1f2;
    }

    .dashboard-stat-label {
        color: #7f8d93;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* --------------------------------------------------------
       PIPELINE
       -------------------------------------------------------- */

    .pipeline-step {
        background: #151b1e;
        border: 1px solid #2a3439;
        border-radius: 10px;
        padding: 15px 17px;
        margin-bottom: 9px;
    }

    .pipeline-number {
        color: #62b5d6;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem;
        margin-bottom: 4px;
    }

    .pipeline-title {
        color: #edf1f2;
        font-weight: 600;
        margin-bottom: 3px;
    }

    .pipeline-description {
        color: #7f8d93;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    /* --------------------------------------------------------
       PRINCIPLES
       -------------------------------------------------------- */

    .principle-card {
        background: #101820;
        border: 1px solid #223142;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    .principle-title {
        color: #62b5d6;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 5px;
    }

    .principle-body {
        color: #93a3ac;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* --------------------------------------------------------
       LIMITATIONS
       -------------------------------------------------------- */

    .limitation-card {
        background: #1a1512;
        border: 1px solid #3a2c15;
        border-radius: 10px;
        padding: 14px 17px;
        margin-bottom: 9px;
    }

    .limitation-title {
        color: #e8a33d;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }

    .limitation-body {
        color: #b8a988;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* --------------------------------------------------------
       DISCLAIMER
       -------------------------------------------------------- */

    .disclaimer {
        background: #1a150a;
        border: 1px solid #4a3718;
        border-radius: 10px;
        padding: 14px 17px;
        color: #a99b7e;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    .disclaimer strong {
        color: #e8a33d;
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #283136;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #edf1f2;
    }

    .sidebar-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #62b5d6;
        margin-bottom: 5px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .footer {
        border-top: 1px solid #283136;
        margin-top: 35px;
        padding-top: 14px;
        color: #66747a;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EVIDENCE ID
# ============================================================

def generate_evidence_id(sha256_hash: str) -> str:
    """
    Generate a stable evidence identifier from the SHA-256 hash.

    Using the hash prefix makes the evidence ID stable across
    Streamlit reruns for the same uploaded file.
    """
    return "IMG-" + sha256_hash[:8].upper()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(CHECKPOINT_PATH):
        return None

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    checkpoint_args = checkpoint.get("args", {})

    if not isinstance(checkpoint_args, dict):
        try:
            checkpoint_args = vars(checkpoint_args)
        except TypeError:
            checkpoint_args = {}

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
        freeze_backbone=False,
        image_size=image_size,
    )

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )

    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "image_size": image_size,
        "backbone": backbone_name,

        "train_acc": checkpoint.get("train_acc"),
        "val_acc": checkpoint.get("val_acc"),
        "test_acc": checkpoint.get("test_acc"),

        "epoch": checkpoint.get("epoch"),

        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
    }


bundle = load_model()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {
    "history": [],
    "last_result": None,
    "last_filename": None,
    "last_sha256": None,
    "report_data": None,
    "report_filename": None,
}

for key, default in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# FORENSIC ANALYSIS
# ============================================================

def run_analysis(
    image: Image.Image,
    filename: str,
    bundle: dict,
    evidence_id: str,
    sha256_hash: str,
    metadata: dict,
    metadata_findings: list,
    manipulation_result: dict = None,
    copy_move_result: dict = None
) -> dict:

    model = bundle["model"]
    device = bundle["device"]
    image_size = bundle["image_size"]

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    config = PreprocessConfig(
        image_size=image_size,
        freq_size=image_size,
        preserve_aspect_ratio=True,
        augment=False,
    )

    preprocessor = ForensicPreprocessor(config)

    # IMPORTANT:
    #
    # Current preprocessing.py returns:
    #
    # display_image,
    # spatial_tensor,
    # frequency_tensor
    #
    display_img, spatial_tensor, freq_tensor = (
        preprocessor.process(
            image,
            augment=False
        )
    )

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    if not isinstance(spatial_tensor, torch.Tensor):
        raise TypeError(
            f"Spatial preprocessing output must be torch.Tensor, "
            f"but got {type(spatial_tensor)}"
        )

    if not isinstance(freq_tensor, torch.Tensor):
        raise TypeError(
            f"Frequency preprocessing output must be torch.Tensor, "
            f"but got {type(freq_tensor)}"
        )

    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    spatial_batch = (
        spatial_tensor
        .unsqueeze(0)
        .to(device)
        .requires_grad_(True)
    )

    freq_batch = (
        freq_tensor
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    gradcam = GradCAMExplainer(
        model,
        target_layer_module=model.gradcam_target_layer
    )

    localizer = ArtifactLocalizer(
        gradcam,
        cam_weight=0.7,
        threshold=0.5
    )

    try:

        (
            loc_result,
            pred_class,
            confidence
        ) = localizer.localize(
            spatial_batch,
            freq_batch,
            display_img
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # CLEAN PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits, _, gates = model(
            spatial_batch,
            freq_batch
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    verdict = CLASS_NAMES[pred_class]

    spatial_trust = float(
        gates[0, 0].item()
    )

    freq_trust = float(
        gates[0, 1].item()
    )

    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------

    notes = []

    if verdict != "Real":

        if freq_trust > spatial_trust:

            notes.append(
                "Frequency-domain evidence contributed more strongly "
                "to the model's AI-generated/manipulated classification."
            )

        else:

            notes.append(
                "Spatial-domain evidence contributed more strongly "
                "to the model's AI-generated/manipulated classification."
            )

        if loc_result.suspicious_ratio > 0.30:

            notes.append(
                "The Grad-CAM analysis highlights a relatively large "
                "portion of the image."
            )

        else:

            notes.append(
                "The Grad-CAM analysis highlights localized image regions."
            )

    else:

        notes.append(
            "The model found stronger evidence consistent with "
            "the real-image class."
        )

        if spatial_trust > freq_trust:

            notes.append(
                "Spatial-domain features contributed more strongly "
                "to the classification."
            )

        else:

            notes.append(
                "Frequency-domain features contributed more strongly "
                "to the classification."
            )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "id": uuid.uuid4().hex[:8].upper(),

        "evidence_id": evidence_id,

        "filename": filename,

        "sha256": sha256_hash,

        "file_size": None,

        "metadata": metadata,

        "metadata_findings": metadata_findings,

        "verdict": verdict,

        "confidence": float(confidence),

        "real_prob": float(
            probabilities[0, 0].item()
        ),

        "ai_prob": float(
            probabilities[0, 1].item()
        ),

        "spatial_trust": spatial_trust,

        "freq_trust": freq_trust,

        "suspicious_ratio": float(
            loc_result.suspicious_ratio
        ),

        "localization_confidence": float(
            loc_result.localization_confidence
        ),

        "notes": notes,

        "display_image": display_img,

        "overlay_image": loc_result.overlay,

        "binary_mask": loc_result.binary_mask,

        "manipulation": manipulation_result or analyze_manipulation(image),

        "copy_move": copy_move_result,

        "date": datetime.now(),
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_html(text: str) -> str:
    """Strip all newlines and excess whitespace so Markdown parses it cleanly as HTML."""
    return " ".join([line.strip() for line in text.splitlines() if line.strip()])


def html(text: str) -> None:
    st.markdown(clean_html(text), unsafe_allow_html=True)


def render_section(
    number: str,
    title: str,
    description: str = ""
):

    description_html = ""

    if description:

        description_html = (
            f'<div class="section-description">'
            f'{description}'
            f'</div>'
        )

    html(
        f"""
        <div class="section-header">
            <div class="section-label">
                {number}
            </div>
            <div class="section-title">
                {title}
            </div>
            {description_html}
        </div>
        """
    )


def render_metric_card(
    label: str,
    value: str
) -> str:

    return clean_html(f"""
    <div class="metric-card">
        <div class="metric-value">
            {value}
        </div>
        <div class="metric-label">
            {label}
        </div>
    </div>
    """)


def render_verdict_card(
    result: dict
) -> str:

    is_real = result["verdict"] == "Real"

    css_class = (
        "verdict-real"
        if is_real
        else "verdict-ai"
    )

    icon = "✓" if is_real else "!"

    return clean_html(f"""
    <div class="verdict-card {css_class}">
        <div class="verdict-label">
            {icon} MODEL CLASSIFICATION
        </div>
        <div class="verdict-name">
            {result["verdict"]}
        </div>
        <div class="verdict-confidence">
            Model confidence:
            <strong>
                {result["confidence"] * 100:.2f}%
            </strong>
        </div>
    </div>
    """)


def render_case_meta(
    result: dict
) -> str:

    return clean_html(f"""
    <div class="case-meta">
        <div class="case-meta-item">
            CASE ID
            <strong>
                {result['id']}
            </strong>
        </div>
        <div class="case-meta-item">
            EVIDENCE ID
            <strong>
                {result['evidence_id']}
            </strong>
        </div>
        <div class="case-meta-item">
            EVIDENCE REFERENCE
            <strong>
                {result['filename']}
            </strong>
        </div>
        <div class="case-meta-item">
            EXAMINED
            <strong>
                {result['date'].strftime('%Y-%m-%d %H:%M:%S')}
            </strong>
        </div>
    </div>
    """)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-label">System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-title">🔎 AIGI Forensics</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if bundle is None:

        st.error(
            "Trained checkpoint not found."
        )

        st.caption(
            f"Expected:\n{CHECKPOINT_PATH}"
        )

    else:

        st.success("Model ready")

        st.markdown(
            f"**Device**  \n`{bundle['device']}`"
        )

        st.markdown(
            f"**Backbone**  \n`{bundle['backbone']}`"
        )

        st.markdown(
            f"**Input size**  \n"
            f"`{bundle['image_size']} × {bundle['image_size']}`"
        )

        if bundle["test_acc"] is not None:

            st.markdown(
                f"**Test accuracy**  \n"
                f"`{bundle['test_acc'] * 100:.2f}%`"
            )

        elif bundle["val_acc"] is not None:

            st.markdown(
                f"**Validation accuracy**  \n"
                f"`{bundle['val_acc'] * 100:.2f}%`"
            )

        if bundle["missing_keys"] or bundle["unexpected_keys"]:

            with st.expander(
                "Checkpoint details"
            ):

                if bundle["missing_keys"]:

                    st.caption("Missing keys:")
                    st.code(
                        str(bundle["missing_keys"])
                    )

                if bundle["unexpected_keys"]:

                    st.caption("Unexpected keys:")
                    st.code(
                        str(bundle["unexpected_keys"])
                    )

    st.divider()

    st.markdown(
        '<div class="sidebar-label">Classification</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Class 0  →  Real"
    )

    st.caption(
        "Class 1  →  AI-Generated / Manipulated"
    )

    st.divider()

    if st.button(
        "Clear Session History",
        use_container_width=True
    ):

        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_filename = None
        st.session_state.last_sha256 = None
        st.session_state.report_data = None
        st.session_state.report_filename = None

        st.rerun()

    st.divider()

    st.caption(
        "Automated forensic triage aid, not a "
        "determinative verdict. Findings should "
        "be corroborated with additional evidence "
        "and expert human review."
    )


# ============================================================
# MAIN HEADER
# ============================================================

validation_text = ""

if (
    bundle is not None
    and bundle.get("test_acc") is not None
):

    validation_text = (
        f'<span class="status-pill">'
        f'<strong>'
        f'{bundle["test_acc"] * 100:.2f}%'
        f'</strong> '
        f'test accuracy'
        f'</span>'
    )

elif (
    bundle is not None
    and bundle.get("val_acc") is not None
):

    validation_text = (
        f'<span class="status-pill">'
        f'<strong>'
        f'{bundle["val_acc"] * 100:.2f}%'
        f'</strong> '
        f'validation accuracy'
        f'</span>'
    )


html(
    f"""
    <div class="app-header">

        <div class="app-kicker">
            DIGITAL IMAGE FORENSICS · AI IMAGE DETECTION
        </div>

        <div class="app-title">
            AI Image Forensics
        </div>

        <div class="app-subtitle">
            A forensic-support system that combines
            spatial and frequency-domain image evidence,
            model explainability, evidence integrity,
            metadata analysis, and structured reporting.
        </div>

        <div class="status-row">

            <span class="status-pill">
                <strong>ConvNeXt-Tiny</strong> spatial
            </span>

            <span class="status-pill">
                <strong>FFT + DWT</strong> frequency
            </span>

            <span class="status-pill">
                <strong>Feature Fusion</strong>
            </span>

            <span class="status-pill">
                <strong>Grad-CAM</strong> explainability
            </span>

            <span class="status-pill">
                <strong>SHA-256</strong> integrity
            </span>

            {validation_text}

        </div>

    </div>
    """
)


# ============================================================
# CHECKPOINT CHECK
# ============================================================

if bundle is None:

    st.error(
        "The trained model checkpoint could not be loaded."
    )

    st.info(
        f"Please verify that this file exists:\n\n"
        f"{CHECKPOINT_PATH}"
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab_analyze, tab_dashboard, tab_about = st.tabs(
    [
        "🔬  Analyze",
        "📊  Dashboard",
        "ℹ️  About",
    ]
)


# ============================================================
# TAB 1 — ANALYZE
# ============================================================

with tab_analyze:

    render_section(
        "01",
        "Evidence Input",
        "Upload the image you want to examine."
    )

    upload_col, preview_col = st.columns(
        [1, 1.4],
        gap="large"
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    with upload_col:

        uploaded_file = st.file_uploader(
            "Upload image",
            type=SUPPORTED_IMAGE_TYPES,
            label_visibility="collapsed"
        )

        st.caption(
            "Supported formats: PNG, JPG, JPEG"
        )

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    image = None
    file_bytes = None
    evidence_id = None
    sha256_hash = None
    metadata = {}
    metadata_findings = []

    # --------------------------------------------------------
    # PROCESS UPLOADED FILE
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:

            # Read original bytes
            file_bytes = uploaded_file.getvalue()

            # Calculate SHA-256
            sha256_hash = calculate_sha256_bytes(
                file_bytes
            )

            # Stable evidence ID
            evidence_id = generate_evidence_id(
                sha256_hash
            )

            # Open original image from bytes
            original_image = Image.open(uploaded_file)

            metadata = extract_metadata(original_image)
            metadata_findings = analyze_metadata(metadata)

            image = original_image.convert("RGB")
            manipulation_result = analyze_manipulation(image)
            copy_move_result = detect_copy_move(image)
            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            try:

                metadata = extract_metadata(
                    original_image
                )

                metadata_findings = analyze_metadata(
                    metadata
                )

            except Exception as metadata_error:

                metadata = {}

                metadata_findings = [
                    "Metadata analysis could not be completed: "
                    + str(metadata_error)
                ]

        except Exception as error:

            st.error(
                f"Unable to process uploaded image: {error}"
            )

            image = None

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    with preview_col:

        if image is not None:

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        else:

            st.info(
                "Upload an image to begin the "
                "forensic analysis."
            )

    # ========================================================
    # EVIDENCE INTEGRITY
    # ========================================================

    if (
        uploaded_file is not None
        and image is not None
        and file_bytes is not None
    ):

        st.divider()

        render_section(
            "02",
            "Evidence Integrity",
            "Identity and integrity information calculated from the original uploaded bytes."
        )

        evidence_col1, evidence_col2 = st.columns(
            2,
            gap="large"
        )

        with evidence_col1:

            st.markdown(
                "**Evidence ID**"
            )

            st.code(
                evidence_id
            )

            st.markdown(
                "**Filename**"
            )

            st.write(
                uploaded_file.name
            )

            st.markdown(
                "**File Size**"
            )

            st.write(
                f"{len(file_bytes):,} bytes"
            )

        with evidence_col2:

            st.markdown(
                "**SHA-256**"
            )

            st.code(
                sha256_hash,
                language="text"
            )

            st.caption(
                "The SHA-256 hash identifies the exact byte sequence "
                "of the uploaded evidence file."
            )

    # ========================================================
    # METADATA FORENSICS
    # ========================================================

    if (
        uploaded_file is not None
        and image is not None
    ):

        st.divider()

        render_section(
            "03",
            "Metadata Forensics",
            "Basic file and EXIF information extracted from the original image."
        )

        meta_col1, meta_col2 = st.columns(
            2,
            gap="large"
        )

        with meta_col1:

            file_format = metadata.get(
                "file_format",
                "Unknown"
            )

            width = metadata.get(
                "width",
                "Unknown"
            )

            height = metadata.get(
                "height",
                "Unknown"
            )

            mode = metadata.get(
                "mode",
                "Unknown"
            )

            st.markdown(
                "**File Format**"
            )

            st.write(
                file_format
            )

            st.markdown(
                "**Image Dimensions**"
            )

            st.write(
                f"{width} × {height}"
            )

            st.markdown(
                "**Color Mode**"
            )

            st.write(
                mode
            )

        with meta_col2:

            exif_data = metadata.get(
                "exif",
                {}
            )

            st.markdown(
                "**EXIF Entries**"
            )

            st.write(
                len(exif_data)
            )

            st.markdown(
                "**Metadata Status**"
            )

            if exif_data:

                st.success(
                    "EXIF metadata detected"
                )

            else:

                st.info(
                    "No EXIF metadata found"
                )

        # ----------------------------------------------------
        # FINDINGS & RAW EXIF
        # ----------------------------------------------------

        if metadata_findings:

            with st.expander(
                "🔎 Metadata Findings",
                expanded=True
            ):

                for finding in metadata_findings:

                    st.write(
                        "• " + str(finding)
                    )

        if exif_data:

            with st.expander(
                "📷 Raw EXIF Tags Table",
                expanded=False
            ):

                exif_table = [
                    {"Tag": str(k), "Value": str(v)}
                    for k, v in exif_data.items()
                ]

                st.dataframe(
                    exif_table,
                    use_container_width=True,
                    hide_index=True
                )

    # ========================================================
    # RUN BUTTON
    # ========================================================

    run_clicked = False

    if (
        uploaded_file is not None
        and image is not None
    ):

        st.divider()

        run_clicked = st.button(
            "🔍  Run Forensic Analysis",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # RUN FORENSIC ANALYSIS
    # ========================================================

    if run_clicked:

        try:

            with st.status(
                "Running forensic pipeline...",
                expanded=True
            ) as status:

                st.write(
                    "① Preserving uploaded evidence bytes..."
                )

                st.write(
                    "② Preprocessing image..."
                )

                st.write(
                    "③ Extracting ConvNeXt-Tiny spatial features..."
                )

                st.write(
                    "④ Performing FFT + Haar-DWT frequency analysis..."
                )

                st.write(
                    "⑤ Fusing spatial and frequency evidence..."
                )

                st.write(
                    "⑥ Generating Grad-CAM explanation..."
                )

                result = run_analysis(
                    image=image,
                    filename=uploaded_file.name,
                    bundle=bundle,
                    evidence_id=evidence_id,
                    sha256_hash=sha256_hash,
                    metadata=metadata,
                    metadata_findings=metadata_findings,
                    manipulation_result=manipulation_result,
                    copy_move_result=copy_move_result,
                )

                result["file_size"] = len(
                    file_bytes
                )

                st.write(
                    "⑦ Preparing forensic report..."
                )

                status.update(
                    label="Forensic analysis complete",
                    state="complete",
                    expanded=False
                )

            # ------------------------------------------------
            # SAVE SESSION RESULT
            # ------------------------------------------------

            st.session_state.history.insert(
                0,
                result
            )

            st.session_state.last_result = result

            st.session_state.last_filename = (
                uploaded_file.name
            )

            st.session_state.last_sha256 = (
                sha256_hash
            )

            st.session_state.report_data = None

            st.session_state.report_filename = None

        except Exception as error:

            st.error(
                "An error occurred during forensic analysis."
            )

            st.exception(error)

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = st.session_state.get(
        "last_result"
    )

    if (
        result is not None
        and uploaded_file is not None
        and st.session_state.last_sha256
        == result.get("sha256")
    ):

        st.divider()

        # ----------------------------------------------------
        # FORENSIC VERDICT
        # ----------------------------------------------------

        render_section(
            "04",
            "Forensic Analysis Result",
            "Classification produced by the spatial-frequency model."
        )

        st.markdown(
            render_case_meta(result),
            unsafe_allow_html=True
        )

        st.markdown(
            render_verdict_card(result),
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        metric1, metric2, metric3 = st.columns(3)

        metric1.markdown(
            render_metric_card(
                "Real Probability",
                f"{result['real_prob'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

        metric2.markdown(
            render_metric_card(
                "AI Probability",
                f"{result['ai_prob'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

        metric3.markdown(
            render_metric_card(
                "Model Confidence",
                f"{result['confidence'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # VISUAL EVIDENCE
        # ----------------------------------------------------

        render_section(
            "05",
            "Visual Evidence",
            "Grad-CAM highlights regions that contributed to the model prediction."
        )

        original_col, overlay_col = st.columns(
            2,
            gap="large"
        )

        with original_col:

            st.markdown(
                '<div class="evidence-title">'
                'Submitted Image'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                result["display_image"],
                use_container_width=True
            )

        with overlay_col:

            st.markdown(
                '<div class="evidence-title">'
                'Grad-CAM Contribution Map'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                result["overlay_image"],
                use_container_width=True
            )

        # ----------------------------------------------------
        # CORROBORATING MODEL EVIDENCE
        # ----------------------------------------------------

        render_section(
            "06",
            "Spatial–Frequency Evidence",
            "Relative contribution of the model's spatial and frequency branches."
        )

        spatial_pct = (
            result["spatial_trust"] * 100
        )

        freq_pct = (
            result["freq_trust"] * 100
        )

        st.progress(
            min(
                max(
                    result["spatial_trust"],
                    0.0
                ),
                1.0
            ),
            text=(
                f"Spatial evidence — "
                f"{spatial_pct:.1f}%"
            )
        )

        st.progress(
            min(
                max(
                    result["freq_trust"],
                    0.0
                ),
                1.0
            ),
            text=(
                f"Frequency evidence — "
                f"{freq_pct:.1f}%"
            )
        )

        f1, f2, f3 = st.columns(3)

        f1.markdown(
            render_metric_card(
                "Suspicious Area",
                f"{result['suspicious_ratio'] * 100:.1f}%"
            ),
            unsafe_allow_html=True
        )

        f2.markdown(
            render_metric_card(
                "Localization Confidence",
                f"{result['localization_confidence'] * 100:.1f}%"
            ),
            unsafe_allow_html=True
        )

        dominant = (
            "Spatial"
            if spatial_pct >= freq_pct
            else "Frequency"
        )

        f3.markdown(
            render_metric_card(
                "Dominant Evidence",
                dominant
            ),
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PHYSICAL & MANIPULATION FORENSICS (ELA & NOISE)
        # ----------------------------------------------------

        render_section(
            "07",
            "Physical & Manipulation Forensics",
            "Error Level Analysis (ELA), Gaussian noise residual, and compression artifacts."
        )

        manip = result.get("manipulation", {})

        if manip:

            m_col1, m_col2 = st.columns(
                2,
                gap="large"
            )

            with m_col1:

                st.markdown(
                    '<div class="evidence-title">'
                    'Error Level Analysis (ELA)'
                    '</div>',
                    unsafe_allow_html=True
                )

                if manip.get("ela_image") is not None:

                    st.image(
                        manip["ela_image"],
                        caption="JPEG Recompression Error (Quality=90)",
                        use_container_width=True
                    )

                st.caption(
                    "Bright regions indicate areas with different compression levels, "
                    "often caused by digital manipulation, splicing, or inpainting."
                )

            with m_col2:

                st.markdown(
                    '<div class="evidence-title">'
                    'Noise Residual Map'
                    '</div>',
                    unsafe_allow_html=True
                )

                if manip.get("noise_image") is not None:

                    st.image(
                        manip["noise_image"],
                        caption="Gaussian Noise Residual (5× Enhanced)",
                        use_container_width=True
                    )

                st.caption(
                    "Visualizes high-frequency noise variance. Natural camera sensors "
                    "leave uniform noise, while generative AI leaves irregular noise distributions."
                )

            # Quantitative forensic metrics
            ela_stats = manip.get("ela_stats", {})
            noise_stats = manip.get("noise_stats", {})
            comp_stats = manip.get("compression", {})

            s1, s2, s3, s4 = st.columns(4)

            s1.markdown(
                render_metric_card(
                    "ELA Mean Error",
                    f"{ela_stats.get('mean_error', 0.0):.2f}"
                ),
                unsafe_allow_html=True
            )

            s2.markdown(
                render_metric_card(
                    "Suspicious ELA Pixels",
                    f"{ela_stats.get('suspicious_pixel_ratio', 0.0) * 100:.1f}%"
                ),
                unsafe_allow_html=True
            )

            s3.markdown(
                render_metric_card(
                    "Noise Residual Std",
                    f"{noise_stats.get('noise_std', 0.0):.2f}"
                ),
                unsafe_allow_html=True
            )

            s4.markdown(
                render_metric_card(
                    "Quantization Tables",
                    str(comp_stats.get("quantization_tables", 0))
                ),
                unsafe_allow_html=True
            )

            # Findings
            if manip.get("findings"):

                with st.expander(
                    "🔎 Physical & Compression Findings",
                    expanded=True
                ):

                    for finding in manip["findings"]:

                        st.write(
                            "• " + str(finding)
                        )

        # ----------------------------------------------------
# COPY-MOVE FORGERY ANALYSIS
# ----------------------------------------------------

        render_section(
            "08",
            "Copy-Move Forgery Analysis",
            "Detects potentially duplicated visual features within the same image."
        )

        copy_move = result.get("copy_move", {})

        if copy_move:

            # Create the two columns BEFORE using them
            cm_col1, cm_col2 = st.columns(2, gap="large")

            with cm_col1:
                st.markdown(
                    '<div class="evidence-title">Submitted Image</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    use_container_width=True
                )

            with cm_col2:
                st.markdown(
                    '<div class="evidence-title">Copy-Move Feature Correspondences</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    copy_move["visualization"],
                    use_container_width=True
                )

    # Metrics
            cm1, cm2 = st.columns(2)

            cm1.markdown(
                render_metric_card(
                    "Feature Correspondences",
                    str(copy_move.get("match_count", 0))
                ),
                unsafe_allow_html=True
            )

            detected_text = (
                "Potential Pattern Detected"
                if copy_move.get("detected", False)
                else "No Strong Pattern"
            )

            cm2.markdown(
                render_metric_card(
                    "Analysis Status",
                    detected_text
                ),
                unsafe_allow_html=True
            )

    # Result message
            if copy_move.get("detected", False):

                st.warning(
                    "Potential copy-move pattern detected. "
                    "Highlighted feature correspondences require "
                    "further forensic examination."
                )

            else:

                st.success(
                    "No strong copy-move pattern was detected "
                    "using the current feature-matching analysis."
                )

    # Findings
            if copy_move.get("findings"):

                with st.expander(
                    "🔎 Copy-Move Analysis Findings",
                    expanded=True
                ):

                    for finding in copy_move["findings"]:
                        st.write("• " + str(finding))

    # Disclaimer
            st.caption(
                "Copy-move analysis provides a forensic indicator only. "
                "Similar local features can occur naturally and do not "
                "independently prove image manipulation."
            )
        # ----------------------------------------------------
        # INVESTIGATOR NOTES
        # ----------------------------------------------------

        render_section(
            "09",
            "Investigator Notes",
            "Automated interpretation of the model and image evidence."
        )

        notes_html = "".join(
            f'<div class="note-item">{note}</div>'
            for note in result["notes"]
        )

        st.markdown(
            f'<div class="note-box">'
            f'{notes_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # SHA-256 REFERENCE
        # ----------------------------------------------------

        render_section(
            "09",
            "Evidence Integrity Reference",
            "Hash value calculated from the original uploaded file."
        )

        st.markdown(
            "**Evidence ID**"
        )

        st.code(
            result["evidence_id"]
        )

        st.markdown(
            "**SHA-256**"
        )

        st.code(
            result["sha256"],
            language="text"
        )

        st.caption(
            "SHA-256 confirms the identity of the analyzed "
            "file bytes. It does not by itself establish "
            "whether the image is authentic."
        )

        # ----------------------------------------------------
        # PDF REPORT
        # ----------------------------------------------------

        render_section(
            "10",
            "Forensic Report",
            "Generate a structured PDF containing the model findings."
        )

        if st.session_state.report_data is None:

            try:

                os.makedirs(
                    REPORT_OUTPUT_DIR,
                    exist_ok=True
                )

                case = ForensicCaseResult(

    # ========================================================
    # CASE / EVIDENCE INFORMATION
    # ========================================================

                    case_id=result["id"],

                    evidence_id=result.get(
                    "evidence_id",
                    ""
                    ),

                    source_filename=result["filename"],

                    file_size=result.get(
                        "file_size"
                    ),

                    sha256=result.get(
                    "sha256",
                    ""
                    ),

                    analysis_date=result.get(
                    "date"
                    ),

    # ========================================================
    # AI CLASSIFICATION
    # ========================================================

                    verdict=result["verdict"],

                    confidence=result["confidence"],

                    real_prob=result.get(
                    "real_prob",
                    0.0
                    ),

                    ai_prob=result.get(
                    "ai_prob",
                    0.0
                    ),

    # ========================================================
    # SPATIAL / FREQUENCY EVIDENCE
    # ========================================================

                    spatial_trust=result["spatial_trust"],

                    freq_trust=result["freq_trust"],

    # ========================================================
    # GRAD-CAM / LOCALIZATION
    # ========================================================

                    suspicious_ratio=result[
                        "suspicious_ratio"
                    ],

                    localization_confidence=result[
                        "localization_confidence"
                    ],

                    display_image=result[
                        "display_image"
                    ],

                    overlay_image=result[
                        "overlay_image"
                    ],

                    binary_mask=result[
                        "binary_mask"
                    ],

    # ========================================================
    # METADATA / EXIF
    # ========================================================

                    metadata=result.get(
                        "metadata",
                        {}
                    ),

                    metadata_findings=result.get(
                        "metadata_findings",
                        []
                    ),

    # ========================================================
    # IMAGE MANIPULATION FORENSICS
    # ========================================================

                    manipulation=result.get(
                        "manipulation",
                        {}
                    ),

    # ========================================================
    # COPY-MOVE FORENSICS
    # ========================================================

                    copy_move=result.get(
                        "copy_move",
                        {}
                    ),

    # ========================================================
    # NOTES
    # ========================================================

                    notes=result.get(
                        "notes",
                        []
                    ),
                )

                reporter = ForensicReportGenerator(
                    output_dir=REPORT_OUTPUT_DIR
                )

                report_path = reporter.generate(
                    case
                )

                with open(
                    report_path,
                    "rb"
                ) as report_file:

                    st.session_state.report_data = (
                        report_file.read()
                    )

                st.session_state.report_filename = (
                    os.path.basename(
                        report_path
                    )
                )

            except Exception as error:

                st.error(
                    f"Unable to generate PDF report: "
                    f"{error}"
                )

        if (
            st.session_state.report_data
            is not None
        ):

            st.download_button(
                "⬇️  Download PDF Forensic Report",
                data=st.session_state.report_data,
                file_name=st.session_state.report_filename,
                mime="application/pdf",
                use_container_width=True,
            )

    # ========================================================
    # EMPTY STATE
    # ========================================================

    elif uploaded_file is None:

        st.divider()

    elif uploaded_file is None:

        st.divider()

        html(
            """
            <div class="evidence-card"
                 style="text-align:center; padding:45px 20px;">

                <div style="font-size:2rem; margin-bottom:10px;">
                    🔎
                </div>

                <div class="evidence-title">
                    Awaiting Evidence
                </div>

                <div class="evidence-subtitle">
                    Upload an image above to begin forensic analysis.
                </div>

            </div>
            """
        )

    elif (
        uploaded_file is not None
        and (
            st.session_state.last_sha256
            != sha256_hash
        )
    ):

        st.info(
            "A new image has been uploaded. "
            "Click **Run Forensic Analysis** to analyze it."
        )


# ============================================================
# TAB 2 — DASHBOARD
# ============================================================

with tab_dashboard:

    render_section(
        "SESSION",
        "Analysis Dashboard",
        "Cases analyzed during the current session."
    )

    history = st.session_state.history

    if not history:

        st.info(
            "No cases have been analyzed yet."
        )

        st.caption(
            "Run an analysis from the Analyze tab "
            "to populate this dashboard."
        )

    else:

        flagged_cases = [
            case
            for case in history
            if case["verdict"] != "Real"
        ]

        real_cases = [
            case
            for case in history
            if case["verdict"] == "Real"
        ]

        average_confidence = (
            sum(
                case["confidence"]
                for case in history
            )
            / len(history)
        )

        c1, c2, c3, c4 = st.columns(4)

        dashboard_values = [
            (
                "Total Cases",
                str(len(history))
            ),
            (
                "AI Flagged",
                str(len(flagged_cases))
            ),
            (
                "Marked Real",
                str(len(real_cases))
            ),
            (
                "Avg Confidence",
                f"{average_confidence * 100:.1f}%"
            ),
        ]

        for column, (
            label,
            value
        ) in zip(
            [c1, c2, c3, c4],
            dashboard_values
        ):

            with column:
                html(
                    f"""
                    <div class="dashboard-stat">

                        <div class="dashboard-stat-value">
                            {value}
                        </div>

                        <div class="dashboard-stat-label">
                            {label}
                        </div>

                    </div>
                    """
                )

        st.markdown("")

        table_col, chart_col = st.columns(
            [1.7, 1],
            gap="large"
        )

        # ----------------------------------------------------
        # CASE LOG
        # ----------------------------------------------------

        with table_col:

            st.markdown(
                "#### Case Log"
            )

            table_rows = [

                {
                    "Case ID": case["id"],

                    "Evidence ID": case.get(
                        "evidence_id",
                        "-"
                    ),

                    "Filename": case["filename"],

                    "Verdict": case["verdict"],

                    "Confidence":
                        f"{case['confidence'] * 100:.1f}%",

                    "Date":
                        case["date"].strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                }

                for case in history
            ]

            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        with chart_col:

            st.markdown(
                "#### Verdict Distribution"
            )

            st.bar_chart(
                {
                    "Real": len(real_cases),
                    "AI-Generated / Manipulated":
                        len(flagged_cases)
                }
            )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab_about:

    render_section(
        "SYSTEM",
        "About the Forensic System",
        "Architecture, methodology, and forensic scope."
    )

    st.write(
        "This system analyzes digital images using "
        "complementary spatial and frequency-domain "
        "evidence. It combines deep-learning "
        "classification, evidence integrity, metadata "
        "analysis, Grad-CAM explainability, and "
        "structured reporting."
    )

    st.markdown("")

    # --------------------------------------------------------
    # FORENSIC PRINCIPLES
    # --------------------------------------------------------

    render_section(
        "PRINCIPLES",
        "Why This Is a Forensic-Support Tool"
    )

    principles = [

        (
            "Evidence integrity",
            "The original uploaded file is hashed using "
            "SHA-256 so the exact analyzed byte sequence "
            "can be identified."
        ),

        (
            "Complementary evidence",
            "Spatial and frequency-domain representations "
            "provide different views of the image and "
            "are combined through the model's fusion mechanism."
        ),

        (
            "Explainability",
            "Grad-CAM highlights image regions that "
            "contributed to the model prediction."
        ),

        (
            "Metadata analysis",
            "Basic file properties and EXIF information "
            "are inspected when available. Missing or "
            "inconsistent metadata is treated as a clue "
            "rather than definitive proof."
        ),

        (
            "Documented findings",
            "The system records a case identifier, "
            "evidence identifier, timestamp, classification, "
            "and supporting model findings."
        ),

        (
            "Human review",
            "Automated findings should be corroborated "
            "with additional forensic evidence and expert "
            "human analysis."
        ),
    ]

    for title, body in principles:

        html(
            f"""
            <div class="principle-card">

                <div class="principle-title">
                    {title}
                </div>

                <div class="principle-body">
                    {body}
                </div>

            </div>
            """
        )

    st.markdown("")

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    render_section(
        "PIPELINE",
        "Forensic Pipeline"
    )

    pipeline = [

        (
            "01",
            "Evidence Identification & Integrity",
            "Records the filename, file size, evidence ID, "
            "and SHA-256 hash of the uploaded evidence."
        ),

        (
            "02",
            "Metadata Forensics",
            "Extracts basic image information and available "
            "EXIF metadata."
        ),

        (
            "03",
            "Image Preprocessing",
            "Standardizes the image and creates spatial "
            "and frequency-domain representations."
        ),

        (
            "04",
            "Spatial Feature Extraction",
            "ConvNeXt-Tiny extracts visual, texture, "
            "and structural features."
        ),

        (
            "05",
            "Frequency-Domain Analysis",
            "FFT and Haar-DWT representations provide "
            "complementary frequency-domain evidence."
        ),

        (
            "06",
            "Feature Fusion",
            "Spatial and frequency evidence are combined "
            "before binary classification."
        ),

        (
            "07",
            "Grad-CAM Explainability",
            "Highlights regions that contributed to "
            "the model prediction."
        ),

        (
            "08",
            "Forensic Reporting",
            "Produces a structured PDF containing "
            "the model findings."
        ),
    ]

    for number, title, description in pipeline:

        html(
            f"""
            <div class="pipeline-step">

                <div class="pipeline-number">
                    MODULE {number}
                </div>

                <div class="pipeline-title">
                    {title}
                </div>

                <div class="pipeline-description">
                    {description}
                </div>

            </div>
            """
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    render_section(
        "MODEL",
        "Model Configuration"
    )

    m1, m2, m3 = st.columns(3)

    m1.markdown(
        render_metric_card(
            "Backbone",
            bundle["backbone"]
        ),
        unsafe_allow_html=True
    )

    m2.markdown(
        render_metric_card(
            "Input",
            f"{bundle['image_size']} px"
        ),
        unsafe_allow_html=True
    )

    if bundle["test_acc"] is not None:

        m3.markdown(
            render_metric_card(
                "Test Accuracy",
                f"{bundle['test_acc'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

    elif bundle["val_acc"] is not None:

        m3.markdown(
            render_metric_card(
                "Validation Accuracy",
                f"{bundle['val_acc'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CLASSES
    # --------------------------------------------------------

    st.markdown("")

    render_section(
        "CLASSES",
        "Classification Output"
    )

    class_col1, class_col2 = st.columns(2)

    with class_col1:

        html(
            """
            <div class="pipeline-step">

                <div class="pipeline-number">
                    CLASS 0
                </div>

                <div class="pipeline-title">
                    Real
                </div>

                <div class="pipeline-description">
                    Image classified as belonging to the
                    real-image class.
                </div>

            </div>
            """
        )

    with class_col2:

        html(
            """
            <div class="pipeline-step">

                <div class="pipeline-number">
                    CLASS 1
                </div>

                <div class="pipeline-title">
                    AI-Generated / Manipulated
                </div>

                <div class="pipeline-description">
                    Image classified as belonging to the
                    AI-generated/manipulated class.
                </div>

            </div>
            """
        )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.markdown("")

    render_section(
        "PERFORMANCE",
        "Checkpoint Performance",
        "Values are read from the loaded checkpoint when available."
    )

    p1, p2, p3 = st.columns(3)

    if bundle["train_acc"] is not None:

        p1.markdown(
            render_metric_card(
                "Training Accuracy",
                f"{bundle['train_acc'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

    if bundle["val_acc"] is not None:

        p2.markdown(
            render_metric_card(
                "Validation Accuracy",
                f"{bundle['val_acc'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

    if bundle["test_acc"] is not None:

        p3.markdown(
            render_metric_card(
                "Test Accuracy",
                f"{bundle['test_acc'] * 100:.2f}%"
            ),
            unsafe_allow_html=True
        )

    st.caption(
        "Performance depends on the dataset, split, "
        "training procedure, and evaluation protocol. "
        "The model output should not be treated as "
        "absolute proof of authenticity or manipulation."
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown("")

    render_section(
        "LIMITATIONS",
        "Known Limitations"
    )

    limitations = [

        (
            "Training data distribution",
            "The training datasets have different source "
            "and resolution characteristics. This can "
            "introduce dataset-related shortcuts that "
            "may affect generalization."
        ),

        (
            "Limited generator diversity",
            "The model has been trained and evaluated on "
            "a particular set of datasets and generators. "
            "Performance on unseen generators or image "
            "distributions is not guaranteed."
        ),

        (
            "External evaluation size",
            "The independent external evaluation used "
            "during development contains a limited number "
            "of images. Therefore, its result should not "
            "be interpreted as a universal real-world "
            "performance estimate."
        ),

        (
            "Metadata limitations",
            "EXIF metadata can be absent, stripped, "
            "modified, or rewritten. Metadata findings "
            "should therefore be treated as supporting "
            "evidence rather than definitive proof."
        ),

        (
            "Grad-CAM limitations",
            "Grad-CAM identifies regions contributing to "
            "the model prediction. A highlighted region "
            "does not by itself prove that the region "
            "was manipulated."
        ),

        (
            "Forensic interpretation",
            "Final forensic conclusions require correlation "
            "with additional evidence such as original "
            "files, source information, device information, "
            "CCTV, provenance records, or other investigative evidence."
        ),
    ]

    for title, body in limitations:

        html(
            f"""
            <div class="limitation-card">

                <div class="limitation-title">
                    {title}
                </div>

                <div class="limitation-body">
                    {body}
                </div>

            </div>
            """
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown("")

    render_section(
        "NOTICE",
        "Forensic Disclaimer"
    )

    html(
        """
        <div class="disclaimer">

        <strong>Important:</strong>
        This system is an automated forensic assistance
        and triage tool. Its predictions should not be
        treated as absolute proof of image authenticity,
        AI generation, or manipulation.

        SHA-256 identifies the analyzed file bytes but
        does not establish authenticity.

        Metadata findings are supporting indicators and
        may be missing or altered.

        Grad-CAM highlights regions that contributed to
        the model prediction but does not prove that those
        pixels were manipulated.

        Automated findings should be corroborated using
        additional forensic evidence and expert human
        analysis before being used in investigative,
        legal, academic, or other high-stakes contexts.

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="footer">

        AI-BASED DIGITAL IMAGE FORENSICS SYSTEM
        &nbsp;·&nbsp;

        ConvNeXt-Tiny + FFT/DWT + Feature Fusion
        + Grad-CAM

        &nbsp;·&nbsp;

        SHA-256 Evidence Integrity + Metadata Analysis

        &nbsp;·&nbsp;

        Forensic Triage Aid — Not a Determinative Verdict

    </div>
    """
)