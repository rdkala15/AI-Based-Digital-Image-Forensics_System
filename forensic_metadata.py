"""
Forensic Metadata & Deep Provenance Analysis Module
---------------------------------------------------

Extracts basic image properties, EXIF metadata, magic bytes binary validation,
trailing payload/steganography detection, and deep text chunk inspection
for AI generation prompts (Stable Diffusion, Midjourney, ComfyUI, DALL-E, NovelAI).

Forensic Roles:
1. Binary Header & Magic Bytes: Verifies file integrity and detects extension spoofing.
2. Trailing Bytes Detection: Identifies hidden payloads appended past the image EOF.
3. AI Generation Parameter Parsing: Extracts conclusive prompts, seeds, samplers, and model names.
4. Camera & Software Provenance: Identifies physical hardware EXIF vs. editing software tags.
5. C2PA / Content Credentials: Scans for digital provenance signatures.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ExifTags


# ============================================================
# MAGIC BYTES SIGNATURES
# ============================================================

MAGIC_SIGNATURES: Dict[str, List[Tuple[bytes, int]]] = {
    "PNG": [(b"\x89PNG\r\n\x1a\n", 0)],
    "JPEG": [(b"\xff\xd8\xff", 0)],
    "WEBP": [(b"RIFF", 0), (b"WEBP", 8)],
    "GIF": [(b"GIF87a", 0), (b"GIF89a", 0)],
    "BMP": [(b"BM", 0)],
    "TIFF": [(b"II*\x00", 0), (b"MM\x00*", 0)],
    "EXECUTABLE_EXE": [(b"MZ", 0)],
    "ZIP_ARCHIVE": [(b"PK\x03\x04", 0)],
    "PDF": [(b"%PDF", 0)],
}


# ============================================================
# 1. BINARY HEADER & MAGIC BYTES INSPECTION
# ============================================================

def inspect_binary_header(
    file_bytes: bytes,
    claimed_filename: str = ""
) -> Dict[str, Any]:
    """
    Inspect raw file bytes for signature matches, extension spoofing,
    and trailing payload data.
    """
    result = {
        "detected_format": "Unknown",
        "is_valid_image": False,
        "is_spoofed": False,
        "spoofed_type": None,
        "claimed_extension": "",
        "header_hex": "",
        "trailing_bytes_count": 0,
        "has_trailing_data": False,
        "anomalies": [],
    }

    if not file_bytes or len(file_bytes) < 16:
        result["anomalies"].append("File byte stream is empty or too short.")
        return result

    result["header_hex"] = file_bytes[:16].hex(" ").upper()

    if claimed_filename:
        ext = Path(claimed_filename).suffix.lower().lstrip(".")
        result["claimed_extension"] = ext

    # Check known magic signatures
    detected = "Unknown"
    for fmt, sigs in MAGIC_SIGNATURES.items():
        match = True
        for sig_bytes, offset in sigs:
            if file_bytes[offset : offset + len(sig_bytes)] != sig_bytes:
                match = False
                break
        if match:
            detected = fmt
            break

    result["detected_format"] = detected

    # Check for executable or archive disguised as image
    if detected in ["EXECUTABLE_EXE", "ZIP_ARCHIVE", "PDF"]:
        result["is_spoofed"] = True
        result["spoofed_type"] = detected
        result["anomalies"].append(
            f"CRITICAL: File masquerades as an image but has a {detected} binary header."
        )

    # Validate image format against claimed extension
    image_formats = ["PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"]
    if detected in image_formats:
        result["is_valid_image"] = True
        ext = result["claimed_extension"]
        if ext:
            ext_map = {
                "png": "PNG",
                "jpg": "JPEG",
                "jpeg": "JPEG",
                "webp": "WEBP",
                "gif": "GIF",
                "bmp": "BMP",
                "tif": "TIFF",
                "tiff": "TIFF",
            }
            expected = ext_map.get(ext)
            if expected and expected != detected:
                result["is_spoofed"] = True
                result["anomalies"].append(
                    f"Extension mismatch: File is .{ext} but actual binary format is {detected}."
                )

    # Check for trailing payload / steganography after image EOF
    trailing_count = 0
    if detected == "PNG":
        iend_pos = file_bytes.find(b"IEND\xaeB`\x82")
        if iend_pos != -1:
            expected_eof = iend_pos + 8
            if len(file_bytes) > expected_eof:
                trailing_count = len(file_bytes) - expected_eof
    elif detected == "JPEG":
        eoi_pos = file_bytes.rfind(b"\xff\xd9")
        if eoi_pos != -1:
            expected_eof = eoi_pos + 2
            if len(file_bytes) > expected_eof:
                trailing_count = len(file_bytes) - expected_eof

    if trailing_count > 0:
        result["trailing_bytes_count"] = trailing_count
        result["has_trailing_data"] = True
        result["anomalies"].append(
            f"Detected {trailing_count:,} trailing bytes appended past image End-Of-File."
        )

    return result


# ============================================================
# 2. DEEP AI GENERATION PROMPT & TEXT CHUNK PARSER
# ============================================================

def parse_automatic1111_parameters(raw_text: str) -> Dict[str, Any]:
    """
    Parse Automatic1111 / Stable Diffusion WebUI parameters block.
    """
    parsed = {
        "engine": "Stable Diffusion / Automatic1111",
        "prompt": "",
        "negative_prompt": "",
        "steps": None,
        "sampler": None,
        "cfg_scale": None,
        "seed": None,
        "model": None,
        "model_hash": None,
        "raw_text": raw_text.strip(),
    }

    lines = raw_text.strip().split("\n")
    if not lines:
        return parsed

    neg_idx = -1
    params_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("Negative prompt:"):
            neg_idx = i
        elif "Steps:" in line and ("Sampler:" in line or "Seed:" in line):
            params_idx = i

    if neg_idx != -1:
        parsed["prompt"] = "\n".join(lines[:neg_idx]).strip()
        if params_idx != -1:
            parsed["negative_prompt"] = "\n".join(lines[neg_idx:params_idx]).replace("Negative prompt:", "").strip()
        else:
            parsed["negative_prompt"] = "\n".join(lines[neg_idx:]).replace("Negative prompt:", "").strip()
    elif params_idx != -1:
        parsed["prompt"] = "\n".join(lines[:params_idx]).strip()
    else:
        parsed["prompt"] = raw_text.strip()

    param_line = lines[params_idx] if params_idx != -1 else raw_text
    tokens = re.findall(r'([\w\s]+):\s*([^,]+)(?:,|$)', param_line)
    for key, val in tokens:
        k = key.strip().lower()
        v = val.strip()
        if k == "steps":
            parsed["steps"] = v
        elif k == "sampler":
            parsed["sampler"] = v
        elif k == "cfg scale":
            parsed["cfg_scale"] = v
        elif k == "seed":
            parsed["seed"] = v
        elif k == "model":
            parsed["model"] = v
        elif k == "model hash":
            parsed["model_hash"] = v

    return parsed


def extract_ai_prompts_and_chunks(
    image: Image.Image,
    file_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """
    Extract hidden AI generation prompts, workflow graphs, and text metadata.
    """
    ai_metadata = {
        "has_ai_prompt": False,
        "generator_tool": None,
        "positive_prompt": None,
        "negative_prompt": None,
        "parameters": {},
        "raw_chunks": {},
        "c2pa_detected": False,
        "xmp_detected": False,
    }

    raw_info = getattr(image, "info", {})
    if not isinstance(raw_info, dict):
        raw_info = {}

    for key, val in raw_info.items():
        if isinstance(val, (str, int, float, bool)):
            ai_metadata["raw_chunks"][str(key)] = str(val)
        elif isinstance(val, bytes):
            try:
                ai_metadata["raw_chunks"][str(key)] = val.decode("utf-8", errors="replace")
            except Exception:
                ai_metadata["raw_chunks"][str(key)] = f"<bytes length={len(val)}>"

    if "parameters" in raw_info:
        param_str = str(raw_info["parameters"])
        parsed = parse_automatic1111_parameters(param_str)
        ai_metadata["has_ai_prompt"] = True
        ai_metadata["generator_tool"] = parsed.get("engine", "Stable Diffusion")
        ai_metadata["positive_prompt"] = parsed.get("prompt")
        ai_metadata["negative_prompt"] = parsed.get("negative_prompt")
        ai_metadata["parameters"] = {
            k: v for k, v in parsed.items()
            if k not in ["engine", "prompt", "negative_prompt", "raw_text"] and v is not None
        }

    if "prompt" in raw_info and not ai_metadata["has_ai_prompt"]:
        prompt_val = raw_info["prompt"]
        try:
            if isinstance(prompt_val, str) and (prompt_val.startswith("{") or prompt_val.startswith("[")):
                comfy_data = json.loads(prompt_val)
                ai_metadata["has_ai_prompt"] = True
                ai_metadata["generator_tool"] = "ComfyUI Node Graph"
                prompts_found = []
                for node_id, node_data in comfy_data.items():
                    if isinstance(node_data, dict):
                        inputs = node_data.get("inputs", {})
                        if "text" in inputs and isinstance(inputs["text"], str) and len(inputs["text"]) > 5:
                            prompts_found.append(inputs["text"])
                if prompts_found:
                    ai_metadata["positive_prompt"] = " | ".join(prompts_found)
        except Exception:
            pass

    if not ai_metadata["has_ai_prompt"]:
        for text_key in ["Description", "Comment", "comment", "Software"]:
            if text_key in raw_info:
                val = str(raw_info[text_key])
                if any(tag in val.lower() for tag in ["novelai", "midjourney", "stable diffusion", "dall-e", "stablediffusion"]):
                    ai_metadata["has_ai_prompt"] = True
                    ai_metadata["generator_tool"] = text_key
                    ai_metadata["positive_prompt"] = val
                    break

    xmp_raw = raw_info.get("XML:com.adobe.xmp", "")
    if isinstance(xmp_raw, bytes):
        xmp_raw = xmp_raw.decode("utf-8", errors="replace")
    xmp_str = str(xmp_raw)

    if "c2pa" in xmp_str.lower() or "contentcredentials" in xmp_str.lower() or "cai:" in xmp_str.lower():
        ai_metadata["c2pa_detected"] = True

    if xmp_str and len(xmp_str) > 20:
        ai_metadata["xmp_detected"] = True

    if file_bytes and not ai_metadata["c2pa_detected"]:
        if b"c2pa" in file_bytes or b"contentcredentials" in file_bytes:
            ai_metadata["c2pa_detected"] = True

    return ai_metadata


# ============================================================
# 3. METADATA EXTRACTION (UNIFIED)
# ============================================================

def extract_metadata(
    image: Image.Image,
    file_bytes: Optional[bytes] = None,
    filename: str = ""
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata, binary header validation, EXIF tags,
    and deep AI prompt parameters.
    """
    binary_result = {}
    if file_bytes:
        binary_result = inspect_binary_header(
            file_bytes,
            claimed_filename=filename
        )
    else:
        binary_result = {
            "detected_format": image.format or "Unknown",
            "is_valid_image": True,
            "is_spoofed": False,
            "has_trailing_data": False,
            "anomalies": [],
        }

    fmt = image.format or binary_result.get("detected_format") or "Unknown"

    metadata = {
        "file_format": fmt,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "exif": {},
        "exif_count": 0,
        "exif_error": None,
        "binary_inspection": binary_result,
        "ai_provenance": {},
    }

    metadata["ai_provenance"] = extract_ai_prompts_and_chunks(
        image,
        file_bytes=file_bytes
    )

    try:
        exif_data = image.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                try:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    else:
                        value = str(value)
                except Exception:
                    value = str(value)
                metadata["exif"][tag_name] = value

            metadata["exif_count"] = len(metadata["exif"])
    except Exception as error:
        metadata["exif_error"] = str(error)

    return metadata


# ============================================================
# 4. METADATA FORENSIC FINDINGS
# ============================================================

def analyze_metadata(metadata: Dict[str, Any]) -> List[str]:
    """
    Identify actionable forensic findings across binary integrity,
    AI generation parameters, and camera metadata.
    """
    findings = []
    exif = metadata.get("exif", {})
    binary = metadata.get("binary_inspection", {})
    ai_prov = metadata.get("ai_provenance", {})

    # 1. AI Prompt Findings
    if ai_prov.get("has_ai_prompt"):
        tool = ai_prov.get("generator_tool", "Generative AI Engine")
        findings.append(
            f"DEFINITIVE GENERATION PROOF: Embedded {tool} metadata discovered inside image text chunks."
        )
        if ai_prov.get("positive_prompt"):
            prompt_preview = str(ai_prov["positive_prompt"])
            if len(prompt_preview) > 90:
                prompt_preview = prompt_preview[:90] + "..."
            findings.append(f"Extracted Generation Prompt: \"{prompt_preview}\"")

        params = ai_prov.get("parameters", {})
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
            findings.append(f"AI Parameters: {param_str}")

    if ai_prov.get("c2pa_detected"):
        findings.append("C2PA / Content Credentials provenance manifest detected.")

    # 2. Binary Header & Spoofing Findings
    if binary.get("is_spoofed"):
        findings.append(
            "SECURITY ALERT: File binary signature does not match claimed file extension."
        )

    for anomaly in binary.get("anomalies", []):
        findings.append(anomaly)

    # 3. Basic Image Properties
    file_format = metadata.get("file_format")
    width = metadata.get("width")
    height = metadata.get("height")
    mode = metadata.get("mode")

    if file_format:
        findings.append(f"Detected image format: {file_format}")
    if width and height:
        findings.append(f"Image dimensions: {width} × {height} pixels")
    if mode:
        findings.append(f"Color mode: {mode}")

    # 4. EXIF Hardware vs Software
    if not exif:
        findings.append("No hardware camera EXIF metadata was found in the image.")
    else:
        findings.append(f"{len(exif)} EXIF metadata entries were detected.")

    if "Make" in exif:
        findings.append(f"Camera manufacturer: {exif['Make']}")
    if "Model" in exif:
        findings.append(f"Camera model: {exif['Model']}")
    if "DateTimeOriginal" in exif:
        findings.append(f"Original capture time: {exif['DateTimeOriginal']}")
    if "Software" in exif:
        findings.append(f"Software recorded in metadata: {exif['Software']}")

    if "GPSInfo" in exif:
        findings.append("GPS geolocation metadata is present.")

    if "Software" in exif:
        software = str(exif["Software"]).lower()
        editing_keywords = [
            "photoshop", "gimp", "lightroom", "paint", "editor",
            "canva", "pixlr", "snapseed", "midjourney", "automatic1111"
        ]
        detected = [kw for kw in editing_keywords if kw in software]
        if detected:
            findings.append(
                f"Metadata indicates image-editing / generative software: {exif['Software']}"
            )

    if metadata.get("exif_error"):
        findings.append(f"EXIF parsing warning: {metadata['exif_error']}")

    return findings


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FORENSIC METADATA & DEEP PROVENANCE MODULE TEST")
    print("=" * 60)

    test_img = Image.new("RGB", (256, 256), (100, 150, 200))
    test_bytes = io.BytesIO()
    test_img.save(test_bytes, format="PNG")
    raw_bytes = test_bytes.getvalue()

    meta = extract_metadata(test_img, file_bytes=raw_bytes, filename="camera_sample.png")
    print("\nBasic Test Image Format:", meta["file_format"])
    print("Binary Valid Image:", meta["binary_inspection"]["is_valid_image"])
    print("Detected Format:", meta["binary_inspection"]["detected_format"])

    findings = analyze_metadata(meta)
    print("\nFindings:")
    for f in findings:
        print(" •", f)

    print("\nFORENSIC METADATA TEST COMPLETED SUCCESSFULLY.")
    print("=" * 60)