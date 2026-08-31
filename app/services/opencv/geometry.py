"""
Geometry utilities for UI bounding boxes.

This module provides low-level geometric calculations used by:

    - alignment.py
    - spacing.py
    - density.py
    - feature extraction
    - rule engine

The module is intentionally independent from OCR, YOLO and the merger.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .ui_types import BoundingBox, BboxGeometry, Detection


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON = 1e-9


# ---------------------------------------------------------------------
# Basic bounding-box geometry
# ---------------------------------------------------------------------

def calculate_bbox_features(
    bbox: BoundingBox,
) -> BboxGeometry:
    """
    Compatibility wrapper for calculate_bbox_geometry().
    """

    return calculate_bbox_geometry(bbox)


def calculate_bbox_geometry(
    bbox: BoundingBox,
) -> BboxGeometry:
    """
    Calculate basic geometric features of a bounding box.

    Features:

        width
        height
        area
        center_x
        center_y
    """

    if not bbox.is_valid():
        raise ValueError(
            f"Invalid bounding box: "
            f"x1={bbox.x1}, "
            f"y1={bbox.y1}, "
            f"x2={bbox.x2}, "
            f"y2={bbox.y2}"
        )

    return BboxGeometry(
        width=bbox.width,
        height=bbox.height,
        area=bbox.area,
        center_x=bbox.center[0],
        center_y=bbox.center[1],
    )


# ---------------------------------------------------------------------
# Basic measurements
# ---------------------------------------------------------------------

def calculate_distance(
    center1: Tuple[float, float],
    center2: Tuple[float, float],
) -> float:
    """
    Calculate Euclidean distance between two points.
    """

    x1, y1 = center1
    x2, y2 = center2

    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


def calculate_horizontal_center_distance(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Absolute distance between bounding-box centers along X axis.
    """

    return abs(
        bbox1.center[0]
        - bbox2.center[0]
    )


def calculate_vertical_center_distance(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Absolute distance between bounding-box centers along Y axis.
    """

    return abs(
        bbox1.center[1]
        - bbox2.center[1]
    )


# ---------------------------------------------------------------------
# Edge gaps
# ---------------------------------------------------------------------

def calculate_horizontal_gap(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculate horizontal gap between two bounding boxes.

    Returns:

        > 0  -> boxes are separated horizontally
        = 0  -> boxes touch or overlap horizontally

    This function does not return negative overlap.
    """

    if bbox1.x2 < bbox2.x1:

        return bbox2.x1 - bbox1.x2

    if bbox2.x2 < bbox1.x1:

        return bbox1.x1 - bbox2.x2

    return 0.0


def calculate_vertical_gap(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculate vertical gap between two bounding boxes.

    Returns:

        > 0  -> boxes are separated vertically
        = 0  -> boxes touch or overlap vertically
    """

    if bbox1.y2 < bbox2.y1:

        return bbox2.y1 - bbox1.y2

    if bbox2.y2 < bbox1.y1:

        return bbox1.y1 - bbox2.y2

    return 0.0


# ---------------------------------------------------------------------
# Directional gaps
# ---------------------------------------------------------------------

def calculate_gap_above(
    upper: BoundingBox,
    lower: BoundingBox,
) -> float:
    """
    Calculate the vertical gap from an upper element to a lower element.

    Returns 0 when the elements overlap or are not vertically ordered.
    """

    if upper.y2 <= lower.y1:

        return lower.y1 - upper.y2

    return 0.0


def calculate_gap_below(
    upper: BoundingBox,
    lower: BoundingBox,
) -> float:
    """
    Calculate the vertical gap below an upper element.
    """

    return calculate_gap_above(
        upper,
        lower,
    )


def calculate_gap_left(
    left: BoundingBox,
    right: BoundingBox,
) -> float:
    """
    Calculate horizontal gap between a left and right element.
    """

    if left.x2 <= right.x1:

        return right.x1 - left.x2

    return 0.0


def calculate_gap_right(
    left: BoundingBox,
    right: BoundingBox,
) -> float:
    """
    Calculate horizontal gap between a left and right element.
    """

    return calculate_gap_left(
        left,
        right,
    )


# ---------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------

def calculate_bbox_overlap_area(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculate intersection area between two bounding boxes.
    """

    x_left = max(
        bbox1.x1,
        bbox2.x1,
    )

    y_top = max(
        bbox1.y1,
        bbox2.y1,
    )

    x_right = min(
        bbox1.x2,
        bbox2.x2,
    )

    y_bottom = min(
        bbox1.y2,
        bbox2.y2,
    )

    width = x_right - x_left
    height = y_bottom - y_top

    if width <= 0 or height <= 0:

        return 0.0

    return width * height


def calculate_bbox_union_area(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculate union area of two bounding boxes.
    """

    overlap = calculate_bbox_overlap_area(
        bbox1,
        bbox2,
    )

    return (
        bbox1.area
        + bbox2.area
        - overlap
    )


def calculate_bbox_iou(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculate Intersection over Union (IoU).
    """

    overlap = calculate_bbox_overlap_area(
        bbox1,
        bbox2,
    )

    if overlap <= 0:

        return 0.0

    union = calculate_bbox_union_area(
        bbox1,
        bbox2,
    )

    if union <= EPSILON:

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            overlap / union,
        ),
    )


# ---------------------------------------------------------------------
# Overlap ratios
# ---------------------------------------------------------------------

def calculate_overlap_ratio(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculate overlap relative to the smaller bounding box.

    This is useful for detecting whether one UI element
    is substantially covered by another.
    """

    overlap = calculate_bbox_overlap_area(
        bbox1,
        bbox2,
    )

    smaller_area = min(
        bbox1.area,
        bbox2.area,
    )

    if smaller_area <= EPSILON:

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            overlap / smaller_area,
        ),
    )


# ---------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------

def bbox_contains(
    outer: BoundingBox,
    inner: BoundingBox,
    tolerance: float = 0.0,
) -> bool:
    """
    Check whether one bounding box contains another.
    """

    return (
        outer.x1 <= inner.x1 + tolerance
        and outer.y1 <= inner.y1 + tolerance
        and outer.x2 >= inner.x2 - tolerance
        and outer.y2 >= inner.y2 - tolerance
    )


def calculate_containment_ratio(
    outer: BoundingBox,
    inner: BoundingBox,
) -> float:
    """
    Calculate the percentage of the inner element's area
    contained inside the outer element.
    """

    if inner.area <= EPSILON:

        return 0.0

    overlap = calculate_bbox_overlap_area(
        outer,
        inner,
    )

    return max(
        0.0,
        min(
            1.0,
            overlap / inner.area,
        ),
    )


# ---------------------------------------------------------------------
# Relative position
# ---------------------------------------------------------------------

def get_relative_position(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> str:
    """
    Determine the dominant relative position of bbox1 with respect
    to bbox2.

    Possible values:

        left
        right
        above
        below
        overlapping
        diagonal
    """

    if calculate_bbox_iou(
        bbox1,
        bbox2,
    ) > 0:

        return "overlapping"

    center1_x, center1_y = bbox1.center
    center2_x, center2_y = bbox2.center

    dx = center1_x - center2_x
    dy = center1_y - center2_y

    if abs(dx) > abs(dy):

        if dx < 0:

            return "left"

        return "right"

    if dy < 0:

        return "above"

    return "below"


# ---------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------

def calculate_pairwise_distances(
    detections: List[Detection],
) -> List[float]:
    """
    Calculate Euclidean distances between detection centers.

    This function is kept for compatibility.

    For UI spacing analysis, prefer using directional gaps
    rather than this metric alone.
    """

    distances: List[float] = []

    for i in range(len(detections)):

        for j in range(
            i + 1,
            len(detections),
        ):

            center1 = detections[i].bbox.center
            center2 = detections[j].bbox.center

            distances.append(
                calculate_distance(
                    center1,
                    center2,
                )
            )

    return distances


# ---------------------------------------------------------------------
# Pairwise geometry
# ---------------------------------------------------------------------

def calculate_pairwise_geometry(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> Dict[str, float | str | bool]:
    """
    Calculate a complete set of useful geometric relationships
    between two bounding boxes.
    """

    return {
        "center_distance": calculate_distance(
            bbox1.center,
            bbox2.center,
        ),
        "horizontal_center_distance":
            calculate_horizontal_center_distance(
                bbox1,
                bbox2,
            ),
        "vertical_center_distance":
            calculate_vertical_center_distance(
                bbox1,
                bbox2,
            ),
        "horizontal_gap":
            calculate_horizontal_gap(
                bbox1,
                bbox2,
            ),
        "vertical_gap":
            calculate_vertical_gap(
                bbox1,
                bbox2,
            ),
        "overlap_area":
            calculate_bbox_overlap_area(
                bbox1,
                bbox2,
            ),
        "iou":
            calculate_bbox_iou(
                bbox1,
                bbox2,
            ),
        "overlap_ratio":
            calculate_overlap_ratio(
                bbox1,
                bbox2,
            ),
        "relative_position":
            get_relative_position(
                bbox1,
                bbox2,
            ),
        "bbox1_contains_bbox2":
            bbox_contains(
                bbox1,
                bbox2,
            ),
        "bbox2_contains_bbox1":
            bbox_contains(
                bbox2,
                bbox1,
            ),
    }


# ---------------------------------------------------------------------
# Image boundary handling
# ---------------------------------------------------------------------

def clip_bbox_to_image(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """
    Clip a bounding box to image boundaries.
    """

    if image_width <= 0 or image_height <= 0:

        raise ValueError(
            "Image dimensions must be positive."
        )

    x1 = max(
        0.0,
        min(
            bbox.x1,
            float(image_width),
        ),
    )

    y1 = max(
        0.0,
        min(
            bbox.y1,
            float(image_height),
        ),
    )

    x2 = max(
        0.0,
        min(
            bbox.x2,
            float(image_width),
        ),
    )

    y2 = max(
        0.0,
        min(
            bbox.y2,
            float(image_height),
        ),
    )

    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


# ---------------------------------------------------------------------
# Detection geometry map
# ---------------------------------------------------------------------

def get_geometries_map(
    detections: List[Detection],
) -> Dict[int, BboxGeometry]:
    """
    Calculate geometry for all valid detections.

    Invalid bounding boxes are ignored.
    """

    geometries: Dict[int, BboxGeometry] = {}

    for detection in detections:

        try:

            geometries[detection.id] = (
                calculate_bbox_geometry(
                    detection.bbox
                )
            )

        except (ValueError, AttributeError):

            continue

    return geometries