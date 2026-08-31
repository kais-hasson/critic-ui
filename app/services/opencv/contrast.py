"""
التعديل المطلوب على: contrast.py
===================================
الفكرة: استخدام font_rgb / background_rgb الدقيقين (الناتجين
من color.py) عند توفرهما بثقة كافية، بدل الاعتماد فقط على
"أغمق بكسل مقابل أفتح بكسل" كتقريب عام.

هذا يتطلب تمرير ColorAnalysis (أو dict عناصره) إلى
calculate_contrast_analysis، لذلك سيتغير التوقيع قليلاً --
main.py سيُعدَّل ليمرر هذه الوسيطة الجديدة (انظر ملف main.py
المعدّل).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from .ui_types import (
    Detection,
    ContrastAnalysis,
    ColorAnalysis,
)

DEFAULT_MIN_CONFIDENCE = 0.30

# النسبة الدنيا لثقة الفصل (split_confidence من color.py)
# قبل ما نثق بـ font_rgb/background_rgb الدقيقين. تحت هذا
# الحد، نرجع لطريقة "أغمق/أفتح بكسل" القديمة كـ fallback آمن.
MIN_SPLIT_CONFIDENCE = 0.03


def relative_luminance(
    rgb: np.ndarray,
) -> float:
    values = (
        rgb.astype(np.float64)
        / 255.0
    )
    values = np.where(
        values <= 0.04045,
        values / 12.92,
        (
            (
                values + 0.055
            )
            / 1.055
        ) ** 2.4,
    )
    return float(
        0.2126 * values[0]
        + 0.7152 * values[1]
        + 0.0722 * values[2]
    )


def contrast_ratio(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
) -> float:
    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return float(
        (lighter + 0.05)
        / (darker + 0.05)
    )


def get_crop(
    image_bgr: np.ndarray,
    detection: Detection,
):
    height, width = image_bgr.shape[:2]
    x1 = max(0, int(detection.bbox.x1))
    y1 = max(0, int(detection.bbox.y1))
    x2 = min(width, int(detection.bbox.x2))
    y2 = min(height, int(detection.bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]


def calculate_element_contrast(
    crop: np.ndarray,
) -> float:
    """Fallback method: darkest pixel vs lightest pixel."""
    if crop is None or crop.size == 0:
        return 1.0
    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )
    pixels = rgb.reshape(-1, 3)
    if len(pixels) < 2:
        return 1.0
    brightness = np.mean(pixels, axis=1)
    dark_index = int(np.argmin(brightness))
    light_index = int(np.argmax(brightness))
    return contrast_ratio(
        pixels[dark_index],
        pixels[light_index],
    )


# ============================================================
# NEW (2026-08-26)
# ============================================================
def calculate_contrast_from_colors(
    font_rgb: List[float],
    background_rgb: List[float],
) -> float:
    """
    Precise contrast using the k-means-derived font/background
    colors from color.py, instead of raw darkest/lightest
    pixel. This is the WCAG-style calculation applied to the
    actual estimated ink color vs. the actual estimated local
    background color of the element.
    """
    font_arr = np.array(font_rgb, dtype=np.float64)
    bg_arr = np.array(background_rgb, dtype=np.float64)
    return contrast_ratio(font_arr, bg_arr)


def calculate_contrast_analysis(
    image_bgr: np.ndarray,
    detections: List[Detection],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    color: Optional[ColorAnalysis] = None,
) -> ContrastAnalysis:
    """
    FIX (2026-08-26): added optional `color` parameter.

    When `color` (the ColorAnalysis produced by
    calculate_color_analysis) is provided, and an element has
    a text-bearing color split with sufficient
    split_confidence, the precise font_rgb/background_rgb pair
    is used instead of the darkest/lightest-pixel
    approximation. Elements without a usable split (non-text
    elements, or low split_confidence) still fall back to the
    original method, so behavior is unchanged when `color` is
    not passed at all -- this keeps the function
    backward-compatible with any existing caller.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ContrastAnalysis(
            status="invalid_image"
        )

    elements: Dict[str, float] = {}
    values: List[float] = []

    for detection in detections:
        if detection.confidence < min_confidence:
            continue
        if not detection.bbox.is_valid():
            continue

        precise_ratio = None

        if color is not None:
            element_color = color.elements.get(
                str(detection.id)
            )
            if (
                element_color is not None
                and element_color.is_text
                and element_color.font_rgb
                and element_color.background_rgb
                and (
                    element_color.split_confidence
                    or 0.0
                )
                >= MIN_SPLIT_CONFIDENCE
            ):
                precise_ratio = (
                    calculate_contrast_from_colors(
                        element_color.font_rgb,
                        element_color.background_rgb,
                    )
                )

        if precise_ratio is not None:
            ratio = precise_ratio
        else:
            crop = get_crop(image_bgr, detection)
            ratio = calculate_element_contrast(crop)

        elements[str(detection.id)] = float(ratio)
        values.append(float(ratio))

    average_ratio = (
        float(np.mean(values)) if values else 0.0
    )

    return ContrastAnalysis(
        average_ratio=average_ratio,
        elements=elements,
        status="ok" if values else "insufficient_data",
    )