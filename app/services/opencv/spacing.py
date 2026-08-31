"""
UI spacing analysis.
"""

from __future__ import annotations

import statistics
from typing import List, Optional, Tuple

from .ui_types import Detection, SpacingAnalysis


DEFAULT_MIN_CONFIDENCE = 0.30
DEFAULT_MIN_TEXT_CONFIDENCE = 0.50
MIN_OVERLAP_RATIO = 0.25


TEXT_CLASSES = {
    "text",
    "heading",
    "paragraph",
    "label",
    "link",
    "title",
    "caption",
}


def _class_name(
    detection: Detection,
) -> str:

    return str(
        getattr(
            detection,
            "type",
            "",
        )
    ).strip().lower()


def _is_text(
    detection: Detection,
) -> bool:

    return _class_name(
        detection
    ) in TEXT_CLASSES


def filter_detections(
    detections: List[Detection],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_text_confidence: float = DEFAULT_MIN_TEXT_CONFIDENCE,
) -> List[Detection]:

    result = []

    for detection in detections:

        if detection.confidence < min_confidence:
            continue

        if not detection.bbox.is_valid():
            continue

        if _is_text(detection):

            if (
                detection.text_confidence is not None
                and detection.text_confidence
                < min_text_confidence
            ):
                continue

        result.append(detection)

    return result


def horizontal_gap(
    first: Detection,
    second: Detection,
) -> float:

    if first.bbox.x2 <= second.bbox.x1:
        return second.bbox.x1 - first.bbox.x2

    if second.bbox.x2 <= first.bbox.x1:
        return first.bbox.x1 - second.bbox.x2

    return 0.0


def vertical_gap(
    first: Detection,
    second: Detection,
) -> float:

    if first.bbox.y2 <= second.bbox.y1:
        return second.bbox.y1 - first.bbox.y2

    if second.bbox.y2 <= first.bbox.y1:
        return first.bbox.y1 - second.bbox.y2

    return 0.0


def vertical_overlap_ratio(
    first: Detection,
    second: Detection,
) -> float:

    overlap = max(
        0.0,
        min(
            first.bbox.y2,
            second.bbox.y2,
        )
        - max(
            first.bbox.y1,
            second.bbox.y1,
        ),
    )

    reference = min(
        first.bbox.height,
        second.bbox.height,
    )

    if reference <= 0:
        return 0.0

    return overlap / reference


def horizontal_overlap_ratio(
    first: Detection,
    second: Detection,
) -> float:

    overlap = max(
        0.0,
        min(
            first.bbox.x2,
            second.bbox.x2,
        )
        - max(
            first.bbox.x1,
            second.bbox.x1,
        ),
    )

    reference = min(
        first.bbox.width,
        second.bbox.width,
    )

    if reference <= 0:
        return 0.0

    return overlap / reference


def find_horizontal_neighbors(
    detections: List[Detection],
) -> List[
    Tuple[Detection, Detection]
]:

    pairs = []

    for current in detections:

        candidates = []

        for other in detections:

            if current.id == other.id:
                continue

            if (
                vertical_overlap_ratio(
                    current,
                    other,
                )
                < MIN_OVERLAP_RATIO
            ):
                continue

            if (
                other.bbox.center[0]
                <= current.bbox.center[0]
            ):
                continue

            distance = (
                other.bbox.x1
                - current.bbox.x2
            )

            candidates.append(
                (distance, other)
            )

        if candidates:

            _, nearest = min(
                candidates,
                key=lambda x: x[0],
            )

            pairs.append(
                (current, nearest)
            )

    return pairs


def find_vertical_neighbors(
    detections: List[Detection],
) -> List[
    Tuple[Detection, Detection]
]:

    pairs = []

    for current in detections:

        candidates = []

        for other in detections:

            if current.id == other.id:
                continue

            if (
                horizontal_overlap_ratio(
                    current,
                    other,
                )
                < MIN_OVERLAP_RATIO
            ):
                continue

            if (
                other.bbox.center[1]
                <= current.bbox.center[1]
            ):
                continue

            distance = (
                other.bbox.y1
                - current.bbox.y2
            )

            candidates.append(
                (distance, other)
            )

        if candidates:

            _, nearest = min(
                candidates,
                key=lambda x: x[0],
            )

            pairs.append(
                (current, nearest)
            )

    return pairs


def calculate_spacing_analysis(
    detections: List[Detection],
    direction: str = "all",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_text_confidence: float = DEFAULT_MIN_TEXT_CONFIDENCE,
) -> SpacingAnalysis:

    filtered = filter_detections(
        detections,
        min_confidence,
        min_text_confidence,
    )

    if len(filtered) < 2:

        return SpacingAnalysis(
            status="insufficient_data"
        )

    values: List[float] = []

    if direction in {
        "horizontal",
        "all",
    }:

        for first, second in (
            find_horizontal_neighbors(
                filtered
            )
        ):

            gap = horizontal_gap(
                first,
                second,
            )

            if gap >= 0:
                values.append(gap)

    if direction in {
        "vertical",
        "all",
    }:

        for first, second in (
            find_vertical_neighbors(
                filtered
            )
        ):

            gap = vertical_gap(
                first,
                second,
            )

            if gap >= 0:
                values.append(gap)

    if not values:

        return SpacingAnalysis(
            status="insufficient_data"
        )

    average = statistics.mean(values)

    std = (
        statistics.pstdev(values)
        if len(values) > 1
        else 0.0
    )

    consistency = (
        1.0
        if average <= 0
        else 1.0 /
        (
            1.0 +
            std / average
        )
    )

    return SpacingAnalysis(
        average=float(average),
        minimum=float(min(values)),
        maximum=float(max(values)),
        std=float(std),
        count=len(values),
        consistency=float(
            max(
                0.0,
                min(
                    1.0,
                    consistency,
                ),
            )
        ),
        status="ok",
    )