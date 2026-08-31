"""
التعديل المطلوب على: main.py
===============================
تغييران:
1) تمرير `color` إلى calculate_contrast_analysis (عشان يستخدم
   font_rgb/background_rgb الدقيقين بدل darkest/lightest pixel).
2) دالة جديدة attach_typography_to_elements() تحقن النتائج
   (font_color, background_color, font_size) داخل كل عنصر من
   عناصر merger_result["elements"] نفسها -- وهذا هو المطلوب
   أصلاً: "إضافة لون كل عنصر إلى العنصر نفسه".
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .ui_types import (
Detection,
UIFeaturesExtended,
merger_elements_to_detections,
serialize,
)

from .spacing import (
calculate_spacing_analysis,
)

from .alignment import (
calculate_alignment_analysis,
)

from .density import (
calculate_density_analysis,
)

from .color import (
calculate_color_analysis,
)

from .contrast import (
calculate_contrast_analysis,
)

# ============================================================
# Input Normalization  (unchanged)
# ============================================================
def normalize_detections(
    detections: Union[
        List[Detection],
        List[Dict[str, Any]],
        None,
    ],
) -> List[Detection]:
    if not detections:
        return []
    first = detections[0]
    if isinstance(first, Detection):
        return detections
    if isinstance(first, dict):
        return merger_elements_to_detections(
            detections
        )
    raise TypeError(
        "detections must contain Detection "
        "objects or dictionaries."
    )


# ============================================================
# NEW (2026-08-26): inject per-element typography/color data
# back into the ORIGINAL merger element dicts, not just into
# features.color.elements. This is what makes font_color,
# background_color and font_size available directly on each
# element, as requested.
# ============================================================
def attach_typography_to_elements(
    elements: List[Dict[str, Any]],
    color_analysis,
) -> List[Dict[str, Any]]:
    """
    `elements` must be the SAME list (same order, same ids)
    that was passed into normalize_detections() /
    merger_elements_to_detections(), since ElementColor.id is
    the index assigned during that conversion.
    """
    updated = []
    for index, element in enumerate(elements):
        element = dict(element)
        element_color = color_analysis.elements.get(
            str(index)
        )
        if element_color is not None:
            if element_color.font_rgb:
                element["font_color"] = {
                    "rgb": element_color.font_rgb,
                    "hsv": element_color.font_hsv,
                }
            if element_color.background_rgb:
                element["background_color"] = {
                    "rgb": element_color.background_rgb,
                    "hsv": element_color.background_hsv,
                }
            if element_color.font_size_px is not None:
                element["font_size_px"] = (
                    element_color.font_size_px
                )
            if element_color.split_confidence is not None:
                element["color_split_confidence"] = (
                    element_color.split_confidence
                )
        updated.append(element)
    return updated


# ============================================================
# Main Extraction
# ============================================================
def extract_ui_features(
    detections: Optional[
        Union[
            List[Detection],
            List[Dict[str, Any]],
        ]
    ] = None,
    image_bgr: Optional[np.ndarray] = None,
    extended: bool = True,
    image_path: Optional[str] = None,
    merger_output: Optional[Dict[str, Any]] = None,
) -> UIFeaturesExtended:
    # --------------------------------------------------------
    # Image loading  (unchanged)
    # --------------------------------------------------------
    if image_bgr is None:
        if image_path is None:
            raise ValueError(
                "Either image_bgr or image_path "
                "must be provided."
            )
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

    height, width = image_bgr.shape[:2]

    # --------------------------------------------------------
    # Get detections from merger output  (unchanged)
    # --------------------------------------------------------
    if detections is None:
        if merger_output is None:
            detections = []
        else:
            detections = merger_output.get("elements", [])

    # --------------------------------------------------------
    # Normalize  (unchanged)
    # --------------------------------------------------------
    detection_objects = normalize_detections(detections)

    # --------------------------------------------------------
    # Spacing / Alignment / Density  (unchanged)
    # --------------------------------------------------------
    spacing = calculate_spacing_analysis(detection_objects)
    alignment = calculate_alignment_analysis(detection_objects)
    density = calculate_density_analysis(
        detection_objects,
        image_width=width,
        image_height=height,
    )

    # --------------------------------------------------------
    # Color  (unchanged call, but color.py now also computes
    # font/background split -- see color.py changes)
    # --------------------------------------------------------
    color = calculate_color_analysis(
        image_bgr,
        detection_objects,
    )

    # --------------------------------------------------------
    # Contrast
    # FIX (2026-08-26): pass `color` so contrast.py can use the
    # precise font_rgb/background_rgb pair when available.
    # --------------------------------------------------------
    contrast = calculate_contrast_analysis(
        image_bgr,
        detection_objects,
        min_confidence=0.30,
        color=color,
    )

    # --------------------------------------------------------
    # Final Features  (unchanged)
    # --------------------------------------------------------
    return UIFeaturesExtended(
        screen={
            "width": int(width),
            "height": int(height),
            "channels": (
                int(image_bgr.shape[2])
                if len(image_bgr.shape) == 3
                else 1
            ),
        },
        element_count=len(detection_objects),
        spacing=spacing,
        alignment=alignment,
        density=density,
        color=color,
        contrast=contrast,
    )


# ============================================================
# JSON helper  (unchanged)
# ============================================================
def extract_ui_features_json(**kwargs) -> Dict[str, Any]:
    features = extract_ui_features(**kwargs)
    return features.to_dict()


# ============================================================
# Merge Original Detection Data + Features
# FIX (2026-08-26): now also injects font_color/background_
# color/font_size_px directly into each element in
# merger_result["elements"], not just into features.color.
# ============================================================
def merge_with_detection_data(
    merger_result: Dict[str, Any],
    features: Union[UIFeaturesExtended, Dict[str, Any]],
) -> Dict[str, Any]:
    if hasattr(features, "to_dict"):
        features_dict = features.to_dict()
        color_analysis = features.color
    else:
        features_dict = serialize(features)
        color_analysis = None  # dict form has no .elements objects

    result = dict(merger_result)
    result["features"] = features_dict

    if color_analysis is not None:
        original_elements = merger_result.get("elements", [])
        result["elements"] = attach_typography_to_elements(
            original_elements,
            color_analysis,
        )

    return result