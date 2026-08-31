"""
UI density analysis.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .ui_types import (
    BoundingBox,
    Detection,
    DensityAnalysis,
)


DEFAULT_MIN_CONFIDENCE = 0.30


def filter_density_detections(
    detections: List[Detection],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> List[Detection]:

    return [

        detection

        for detection in detections

        if (
            detection.confidence
            >= min_confidence
            and detection.bbox.is_valid()
        )
    ]


def calculate_union_area(
    bboxes: List[BoundingBox],
) -> float:

    if not bboxes:
        return 0.0

    min_x = int(
        min(
            bbox.x1
            for bbox in bboxes
        )
    )

    min_y = int(
        min(
            bbox.y1
            for bbox in bboxes
        )
    )

    max_x = int(
        max(
            bbox.x2
            for bbox in bboxes
        )
    )

    max_y = int(
        max(
            bbox.y2
            for bbox in bboxes
        )
    )

    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return 0.0

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for bbox in bboxes:

        x1 = int(bbox.x1) - min_x
        y1 = int(bbox.y1) - min_y
        x2 = int(bbox.x2) - min_x
        y2 = int(bbox.y2) - min_y

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1:

            mask[
                y1:y2,
                x1:x2,
            ] = 1

    return float(
        np.sum(mask)
    )


def calculate_density_analysis(
    detections: List[Detection],
    image_width: int = None,
    image_height: int = None,
    screen_width: int = None,
    screen_height: int = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> DensityAnalysis:

    width = (
        image_width
        if image_width is not None
        else screen_width
    )

    height = (
        image_height
        if image_height is not None
        else screen_height
    )

    if (
        width is None
        or height is None
        or width <= 0
        or height <= 0
    ):

        return DensityAnalysis(
            status="invalid_screen"
        )

    filtered = filter_density_detections(
        detections,
        min_confidence,
    )

    screen_area = float(
        width * height
    )

    occupied_area = calculate_union_area(
        [
            detection.bbox
            for detection in filtered
        ]
    )

    density = (
        occupied_area
        / screen_area
    )

    density = max(
        0.0,
        min(
            1.0,
            density,
        ),
    )

    return DensityAnalysis(
        density=float(density),
        occupied_area=float(
            occupied_area
        ),
        screen_area=float(
            screen_area
        ),
        empty_ratio=float(
            1.0 - density
        ),
        element_count=len(
            filtered
        ),
        status="ok",
    )


def calculate_density(
    detections: List[Detection],
    image_width: int,
    image_height: int,
) -> DensityAnalysis:

    return calculate_density_analysis(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )