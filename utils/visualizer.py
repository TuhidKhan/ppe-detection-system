"""
==============================================================================
PPE Detection Project - Visualizer Module
==============================================================================
This module handles all visual rendering:
  - Draws colored bounding boxes around detected objects:
      • Green for 'helmet' (Safe compliance)
      • Red for 'head' (Critical safety violation)
      • Cyan/Blue for 'person' (Worker body)
  - Draws semi-transparent badge headers with compliance status.
  - Converts images between OpenCV (BGR), PIL, and Base64 strings for API delivery.
==============================================================================
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Optional

# Color Palette in BGR format (OpenCV uses BGR, not RGB)
COLOR_MAP = {
    "helmet": (34, 197, 94),     # Vibrant Green (Safe)
    "head": (68, 68, 239),       # Vibrant Red (Violation)
    "person": (246, 130, 59),    # Blue/Cyan (Worker context)
}
DEFAULT_COLOR = (200, 200, 200)   # Neutral Gray for unknown classes


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draws styled bounding boxes and label tags onto an OpenCV image.

    Parameters:
        image (np.ndarray): The source image in OpenCV BGR format.
        detections (list): List of detected items, each with:
                           {
                               "class_name": str,
                               "confidence": float,
                               "box": [x1, y1, x2, y2]
                           }
        line_thickness (int): Thickness of the bounding box borders.

    Returns:
        np.ndarray: Annotated copy of the image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Calculate dynamic font scale based on image dimensions
    font_scale = max(0.45, min(w, h) / 1000.0)
    font_thickness = max(1, int(font_scale * 1.8))

    for det in detections:
        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)
        box = det.get("box", [0, 0, 0, 0])

        x1, y1, x2, y2 = [int(v) for v in box]

        # Clamp box coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Get class color (default to gray if unknown)
        color = COLOR_MAP.get(class_name, DEFAULT_COLOR)

        # 1. Draw Bounding Box Rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # 2. Prepare Label Text
        if class_name == "head":
            label_text = f"NO HELMET {int(confidence * 100)}%"
        elif class_name == "helmet":
            label_text = f"Helmet {int(confidence * 100)}%"
        else:
            label_text = f"{class_name.capitalize()} {int(confidence * 100)}%"

        # 3. Calculate label text size for background badge
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Determine label position (place above box, or inside if too close to top)
        label_y1 = max(0, y1 - text_h - 8)
        label_y2 = y1 if y1 - text_h - 8 >= 0 else y1 + text_h + 8
        label_x2 = min(w - 1, x1 + text_w + 10)

        # 4. Draw filled background pill for text readability
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)

        # 5. Draw white text inside the label pill
        text_baseline_y = label_y2 - 4 if y1 - text_h - 8 >= 0 else label_y2 - 4
        cv2.putText(
            annotated,
            label_text,
            (x1 + 5, text_baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return annotated


def draw_status_banner(
    image: np.ndarray,
    compliance_report: Dict[str, Any]
) -> np.ndarray:
    """
    Draws a modern status banner at the top of the image showing:
      - Overall compliance badge (SAFE vs VIOLATION)
      - Summary counts of helmets, missing helmets, and total workers.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Banner dimensions
    banner_height = max(50, int(h * 0.08))

    # Banner background colors based on safety level (in BGR)
    status_level = compliance_report.get("status_level", "neutral")
    if status_level == "safe":
        bg_color = (34, 197, 94)      # Green
        status_text = "[SAFE] ALL WORKERS COMPLIANT"
    elif status_level == "danger":
        bg_color = (68, 68, 239)      # Red
        status_text = "[ALERT] SAFETY VIOLATION DETECTED"
    elif status_level == "warning":
        bg_color = (11, 158, 245)     # Amber/Orange
        status_text = "[CAUTION] GEAR UNCONFIRMED"
    else:
        bg_color = (75, 85, 99)       # Neutral Dark Slate
        status_text = "[INFO] NO WORKERS DETECTED"

    # 1. Create a transparent overlay for the top banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), bg_color, -1)

    # Blend overlay with original image (alpha = 0.85 for slight transparency)
    cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)

    # 2. Format banner text
    counts = compliance_report.get("counts", {})
    helmets = counts.get("helmet", 0)
    heads = counts.get("head", 0)
    rate = compliance_report.get("compliance_rate", 100.0)

    sub_text = f"Compliance: {rate}% | Helmets: {helmets} | Violations: {heads}"

    font_scale_title = max(0.55, min(w, h) / 900.0)
    font_scale_sub = max(0.40, min(w, h) / 1300.0)

    # 3. Render Status and Summary text
    cv2.putText(
        annotated,
        status_text,
        (15, int(banner_height * 0.45)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale_title,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        annotated,
        sub_text,
        (15, int(banner_height * 0.85)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale_sub,
        (240, 240, 240),
        1,
        cv2.LINE_AA
    )

    return annotated


def encode_image_to_base64(image: np.ndarray, format_ext: str = ".jpg") -> str:
    """
    Encodes an OpenCV image to a base64 data URI string.
    This allows the image to be directly embedded in JSON responses for web clients.
    """
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        raise ValueError("Failed to encode image to buffer.")

    base64_str = base64.b64encode(buffer).decode("utf-8")
    mime_type = "image/jpeg" if format_ext.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime_type};base64,{base64_str}"


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes (e.g. from an HTTP upload) into an OpenCV BGR numpy array.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the provided image file.")
    return image
