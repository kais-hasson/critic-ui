"""
Canonical data types for the UI Critic feature extraction pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Serialization
# ============================================================
def serialize(value: Any) -> Any:
    """Convert arbitrary values to JSON-compatible values."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return serialize(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ============================================================
# Bounding Box
# ============================================================
@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    def is_valid(self) -> bool:
        return (
            self.x2 > self.x1
            and self.y2 > self.y1
            and self.area > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "width": float(self.width),
            "height": float(self.height),
            "area": float(self.area),
            "center": [
                float(self.center[0]),
                float(self.center[1]),
            ],
        }


# ============================================================
# Detection
# ============================================================
@dataclass
class Detection:
    id: int
    type: str
    confidence: float
    bbox: BoundingBox
    text: Optional[str] = None
    text_confidence: Optional[float] = None
    label: Optional[str] = None
    label_confidence: Optional[float] = None
    label_match_score: Optional[float] = None
    value: Optional[str] = None
    value_confidence: Optional[float] = None
    value_match_score: Optional[float] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def class_name(self) -> str:
        return self.type

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": int(self.id),
            "class": str(self.type),
            "confidence": float(self.confidence),
            "bbox": [
                float(self.bbox.x1),
                float(self.bbox.y1),
                float(self.bbox.x2),
                float(self.bbox.y2),
            ],
            "width": float(self.bbox.width),
            "height": float(self.bbox.height),
            "area": float(self.bbox.area),
            "center": [
                float(self.bbox.center[0]),
                float(self.bbox.center[1]),
            ],
        }
        if self.text is not None:
            result["text"] = self.text
        if self.text_confidence is not None:
            result["text_confidence"] = float(
                self.text_confidence
            )
        if self.label is not None:
            result["label"] = self.label
        if self.label_confidence is not None:
            result["label_confidence"] = float(
                self.label_confidence
            )
        if self.label_match_score is not None:
            result["label_match_score"] = float(
                self.label_match_score
            )
        if self.value is not None:
            result["value"] = self.value
        if self.value_confidence is not None:
            result["value_confidence"] = float(
                self.value_confidence
            )
        if self.value_match_score is not None:
            result["value_match_score"] = float(
                self.value_match_score
            )
        if self.metadata:
            result["metadata"] = serialize(
                self.metadata
            )
        return result


# ============================================================
# Merger -> Detection
# ============================================================
def merger_elements_to_detections(
    elements: List[Dict[str, Any]],
) -> List[Detection]:
    detections: List[Detection] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        bbox_data = element.get(
            "bbox",
            [0, 0, 0, 0],
        )
        if (
            not isinstance(
                bbox_data,
                (list, tuple),
            )
            or len(bbox_data) != 4
        ):
            continue
        try:
            bbox = BoundingBox(
                float(bbox_data[0]),
                float(bbox_data[1]),
                float(bbox_data[2]),
                float(bbox_data[3]),
            )
        except (TypeError, ValueError):
            continue
        if not bbox.is_valid():
            continue
        class_name = (
            element.get("class")
            or element.get("type")
            or element.get("class_name")
            or "Unknown"
        )
        try:
            confidence = float(
                element.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            confidence = 0.0
        excluded_keys = {
            "class",
            "type",
            "class_name",
            "confidence",
            "bbox",
            "width",
            "height",
            "area",
            "center",
            "text",
            "text_confidence",
            "label",
            "label_confidence",
            "label_match_score",
            "value",
            "value_confidence",
            "value_match_score",
        }
        detection = Detection(
            id=index,
            type=str(class_name),
            confidence=confidence,
            bbox=bbox,
            text=element.get("text"),
            text_confidence=element.get(
                "text_confidence"
            ),
            label=element.get("label"),
            label_confidence=element.get(
                "label_confidence"
            ),
            label_match_score=element.get(
                "label_match_score"
            ),
            value=element.get("value"),
            value_confidence=element.get(
                "value_confidence"
            ),
            value_match_score=element.get(
                "value_match_score"
            ),
            metadata={
                key: value
                for key, value in element.items()
                if key not in excluded_keys
            },
        )
        detections.append(detection)
    return detections


# ============================================================
# Spacing
# ============================================================
@dataclass
class SpacingAnalysis:
    average: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    std: Optional[float] = None
    count: int = 0
    consistency: Optional[float] = None
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "average": self.average,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "std": self.std,
            "count": int(self.count),
            "consistency": self.consistency,
            "status": self.status,
        }


# ============================================================
# Alignment
# ============================================================
@dataclass
class AlignmentAnalysis:
    score: float = 0.0
    horizontal_score: float = 0.0
    vertical_score: float = 0.0
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "horizontal_score": float(
                self.horizontal_score
            ),
            "vertical_score": float(
                self.vertical_score
            ),
            "status": self.status,
        }


# ============================================================
# Density
# ============================================================
@dataclass
class DensityAnalysis:
    density: Optional[float] = None
    occupied_area: float = 0.0
    screen_area: float = 0.0
    empty_ratio: Optional[float] = None
    element_count: int = 0
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "density": self.density,
            "occupied_area": float(
                self.occupied_area
            ),
            "screen_area": float(
                self.screen_area
            ),
            "empty_ratio": self.empty_ratio,
            "element_count": int(
                self.element_count
            ),
            "status": self.status,
        }


# ============================================================
# Per Element Color
# FIX (2026-08-26): added font_rgb, font_hsv, background_rgb,
# background_hsv, split_confidence, font_size_px, is_text.
# ============================================================
@dataclass
class ElementColor:
    id: int
    average_rgb: List[float] = field(
        default_factory=list
    )
    average_hsv: List[float] = field(
        default_factory=list
    )
    dominant_rgb: List[float] = field(
        default_factory=list
    )
    status: str = "unknown"

    # ------------------------------------------------------
    # NEW (2026-08-26): typography-related fields.
    # ------------------------------------------------------
    font_rgb: List[float] = field(
        default_factory=list
    )
    font_hsv: List[float] = field(
        default_factory=list
    )
    background_rgb: List[float] = field(
        default_factory=list
    )
    background_hsv: List[float] = field(
        default_factory=list
    )
    split_confidence: Optional[float] = None
    font_size_px: Optional[float] = None
    is_text: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": int(self.id),
            "average_rgb": [
                float(x)
                for x in self.average_rgb
            ],
            "average_hsv": [
                float(x)
                for x in self.average_hsv
            ],
            "dominant_rgb": [
                float(x)
                for x in self.dominant_rgb
            ],
            "status": self.status,
            "is_text": bool(self.is_text),
        }
        if self.font_rgb:
            result["font_rgb"] = [
                float(x) for x in self.font_rgb
            ]
        if self.font_hsv:
            result["font_hsv"] = [
                float(x) for x in self.font_hsv
            ]
        if self.background_rgb:
            result["background_rgb"] = [
                float(x)
                for x in self.background_rgb
            ]
        if self.background_hsv:
            result["background_hsv"] = [
                float(x)
                for x in self.background_hsv
            ]
        if self.split_confidence is not None:
            result["split_confidence"] = float(
                self.split_confidence
            )
        if self.font_size_px is not None:
            result["font_size_px"] = float(
                self.font_size_px
            )
        return result


# ============================================================
# Color
# ============================================================
@dataclass
class ColorAnalysis:
    background_rgb: List[float] = field(
        default_factory=list
    )
    average_rgb: List[float] = field(
        default_factory=list
    )
    average_hsv: List[float] = field(
        default_factory=list
    )
    elements: Dict[
        str,
        ElementColor,
    ] = field(
        default_factory=dict
    )
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "background_rgb": [
                float(x)
                for x in self.background_rgb
            ],
            "average_rgb": [
                float(x)
                for x in self.average_rgb
            ],
            "average_hsv": [
                float(x)
                for x in self.average_hsv
            ],
            "elements": {
                str(element_id): serialize(
                    element
                )
                for element_id, element
                in self.elements.items()
            },
            "status": self.status,
        }


# ============================================================
# Contrast
# ============================================================
@dataclass
class ContrastAnalysis:
    average_ratio: float = 0.0
    elements: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "average_ratio": float(
                self.average_ratio
            ),
            "elements": serialize(
                self.elements
            ),
            "status": self.status,
        }


# ============================================================
# Complete Features
# ============================================================
@dataclass
class UIFeaturesExtended:
    screen: Dict[str, Any]
    element_count: int
    spacing: SpacingAnalysis
    alignment: AlignmentAnalysis
    density: DensityAnalysis
    color: ColorAnalysis
    contrast: ContrastAnalysis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen": serialize(self.screen),
            "element_count": int(
                self.element_count
            ),
            "spacing": serialize(
                self.spacing
            ),
            "alignment": serialize(
                self.alignment
            ),
            "density": serialize(
                self.density
            ),
            "color": serialize(
                self.color
            ),
            "contrast": serialize(
                self.contrast
            ),
        }
@dataclass
class BboxGeometry:
    width: float
    height: float
    area: float
    center_x: float
    center_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": float(self.width),
            "height": float(self.height),
            "area": float(self.area),
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
        }



UIFeaturesCompact = UIFeaturesExtended