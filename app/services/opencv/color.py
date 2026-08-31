"""
UI color analysis.
Provides:
- Screen average RGB
- Screen average HSV
- Background RGB
- Average RGB for every detected element
- Average HSV for every detected element
- Dominant RGB for every detected element
- NEW (2026-08-26): font color / local background color split
  and estimated font size for text-bearing elements.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ui_types import (
    Detection,
    ColorAnalysis,
    ElementColor,
)


# ============================================================
# Basic color helpers  (unchanged from original)
# ============================================================
def bgr_to_rgb_list(
    value,
) -> List[float]:
    return [
        float(value[2]),
        float(value[1]),
        float(value[0]),
    ]


def calculate_average_rgb(
    image_bgr: np.ndarray,
) -> List[float]:
    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        return []
    average = np.mean(
        image_bgr,
        axis=(0, 1),
    )
    return bgr_to_rgb_list(
        average
    )


def calculate_average_hsv(
    image_bgr: np.ndarray,
) -> List[float]:
    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        return []
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )
    average = np.mean(
        hsv,
        axis=(0, 1),
    )
    return [
        float(average[0]),
        float(average[1]),
        float(average[2]),
    ]


def calculate_background_rgb(
    image_bgr: np.ndarray,
) -> List[float]:
    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        return []
    height, width = image_bgr.shape[:2]
    border = max(
        1,
        int(
            min(
                height,
                width,
            ) * 0.02
        ),
    )
    samples = np.concatenate(
        [
            image_bgr[
                :border,
                :,
            ].reshape(-1, 3),
            image_bgr[
                -border:,
                :,
            ].reshape(-1, 3),
            image_bgr[
                :,
                :border,
            ].reshape(-1, 3),
            image_bgr[
                :,
                -border:,
            ].reshape(-1, 3),
        ],
        axis=0,
    )
    median = np.median(
        samples,
        axis=0,
    )
    return bgr_to_rgb_list(
        median
    )


def calculate_dominant_rgb(
    image_bgr: np.ndarray,
) -> List[float]:
    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        return []
    pixels = image_bgr.reshape(
        -1,
        3,
    )
    if len(pixels) > 5000:
        indices = np.random.choice(
            len(pixels),
            5000,
            replace=False,
        )
        pixels = pixels[indices]
    pixels = np.float32(
        pixels
    )
    k = min(
        3,
        len(pixels),
    )
    if k <= 0:
        return []
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        ),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(
        labels.flatten()
    )
    dominant_index = int(
        np.argmax(counts)
    )
    dominant_bgr = centers[
        dominant_index
    ]
    return bgr_to_rgb_list(
        dominant_bgr
    )


def extract_element_image(
    image_bgr: np.ndarray,
    detection: Detection,
):
    height, width = image_bgr.shape[:2]
    x1 = max(
        0,
        int(
            round(
                detection.bbox.x1
            )
        ),
    )
    y1 = max(
        0,
        int(
            round(
                detection.bbox.y1
            )
        ),
    )
    x2 = min(
        width,
        int(
            round(
                detection.bbox.x2
            )
        ),
    )
    y2 = min(
        height,
        int(
            round(
                detection.bbox.y2
            )
        ),
    )
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[
        y1:y2,
        x1:x2,
    ]
    if crop.size == 0:
        return None
    return crop


# ============================================================
# NEW (2026-08-26): typography helpers
# ============================================================
TEXT_BEARING_CLASSES = {
    "text",
    "heading",
    "link",
    "button",
    "utility button",
    "text input",
    "badge",
    "tab",
}


def is_text_bearing(
    detection: Detection,
) -> bool:
    """
    An element is considered text-bearing either because the
    merger already attached OCR text to it, or because its
    class is one that typically renders text even without a
    successful OCR match (e.g. an unmatched "Text" box).
    """
    if detection.text:
        return True
    return (
        str(detection.type).lower()
        in TEXT_BEARING_CLASSES
    )


FONT_SIZE_HEIGHT_RATIO = 0.80
# Empirical ratio: rendered glyph height is typically ~75-85%
# of the CSS/point font size for common UI fonts. This is an
# APPROXIMATION good enough for relative comparisons (e.g. "is
# this heading meaningfully larger than body text?").


def estimate_font_size(
    detection: Detection,
) -> Optional[float]:
    if not is_text_bearing(detection):
        return None
    if detection.bbox.height <= 0:
        return None
    return float(
        detection.bbox.height
        * FONT_SIZE_HEIGHT_RATIO
    )


def split_text_background(
    crop: np.ndarray,
) -> Tuple[
    List[float],
    List[float],
    float,
]:
    """
    Separates a text-bearing element's crop into an estimated
    font color and local background color using k-means with
    k=2.

    Returns (font_rgb, background_rgb, split_confidence).

    Assumption: text pixels are the MINORITY within a text
    element's bounding box (glyphs are thin strokes on a much
    larger background area). The smaller k-means cluster is
    therefore treated as the font color, and the larger
    cluster as the local background color.

    split_confidence is the fraction of pixels belonging to
    the minority (font) cluster. Low values (near 0) or
    high values (near 0.5) both signal an unreliable split.
    """
    if crop is None or crop.size == 0:
        return [], [], 0.0

    pixels = crop.reshape(-1, 3)
    if len(pixels) < 4:
        return [], [], 0.0

    if len(pixels) > 3000:
        indices = np.random.choice(
            len(pixels),
            3000,
            replace=False,
        )
        pixels = pixels[indices]

    pixels_f = np.float32(pixels)

    _, labels, centers = cv2.kmeans(
        pixels_f,
        2,
        None,
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        ),
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()
    counts = np.bincount(
        labels,
        minlength=2,
    )
    total = counts.sum()
    if total == 0:
        return [], [], 0.0

    minority_index = int(np.argmin(counts))
    majority_index = int(np.argmax(counts))

    font_bgr = centers[minority_index]
    background_bgr = centers[majority_index]

    split_confidence = float(
        counts[minority_index] / total
    )

    return (
        bgr_to_rgb_list(font_bgr),
        bgr_to_rgb_list(background_bgr),
        split_confidence,
    )


def rgb_to_hsv_list(
    rgb: List[float],
) -> List[float]:
    if not rgb:
        return []
    pixel = np.uint8(
        [[[
            int(rgb[2]),
            int(rgb[1]),
            int(rgb[0]),
        ]]]
    )
    hsv = cv2.cvtColor(
        pixel,
        cv2.COLOR_BGR2HSV,
    )[0][0]
    return [
        float(hsv[0]),
        float(hsv[1]),
        float(hsv[2]),
    ]


# ============================================================
# Main entry point
# FIX (2026-08-26): element loop now also computes font color,
# local background color, split confidence, and estimated font
# size for text-bearing elements.
# ============================================================
def calculate_color_analysis(
    image_bgr: np.ndarray,
    detections: List[Detection],
) -> ColorAnalysis:
    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        return ColorAnalysis(
            status="invalid_image"
        )

    element_colors: Dict[
        str,
        ElementColor,
    ] = {}

    for detection in detections:
        if not detection.bbox.is_valid():
            continue

        crop = extract_element_image(
            image_bgr,
            detection,
        )
        if crop is None:
            element_colors[
                str(detection.id)
            ] = ElementColor(
                id=detection.id,
                status="invalid_crop",
            )
            continue

        text_bearing = is_text_bearing(
            detection
        )

        element_color = ElementColor(
            id=detection.id,
            average_rgb=calculate_average_rgb(
                crop
            ),
            average_hsv=calculate_average_hsv(
                crop
            ),
            dominant_rgb=calculate_dominant_rgb(
                crop
            ),
            status="ok",
            is_text=text_bearing,
        )

        if text_bearing:
            (
                font_rgb,
                background_rgb,
                split_confidence,
            ) = split_text_background(crop)

            element_color.font_rgb = font_rgb
            element_color.background_rgb = (
                background_rgb
            )
            element_color.split_confidence = (
                split_confidence
            )
            if font_rgb:
                element_color.font_hsv = (
                    rgb_to_hsv_list(font_rgb)
                )
            if background_rgb:
                element_color.background_hsv = (
                    rgb_to_hsv_list(
                        background_rgb
                    )
                )
            element_color.font_size_px = (
                estimate_font_size(detection)
            )

        element_colors[
            str(detection.id)
        ] = element_color

    return ColorAnalysis(
        background_rgb=calculate_background_rgb(
            image_bgr
        ),
        average_rgb=calculate_average_rgb(
            image_bgr
        ),
        average_hsv=calculate_average_hsv(
            image_bgr
        ),
        elements=element_colors,
        status="ok",
    )