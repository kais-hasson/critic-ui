"""
UI Merger
=========
Merges ScreenParser (YOLO) detections with OCRParser results.

Input:
    YOLO:
        {
            "detections": [...]
        }

    OCR:
        {
            "texts": [...]
        }

Output:
    {
        "elements": [...],
        "statistics": {...}
    }
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Utility functions
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def bbox_area(bbox: List[float]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0

    x1, y1, x2, y2 = map(safe_float, bbox[:4])

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    if not bbox or len(bbox) < 4:
        return 0.0, 0.0

    x1, y1, x2, y2 = map(safe_float, bbox[:4])

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def bbox_size(bbox: List[float]) -> Tuple[float, float]:
    if not bbox or len(bbox) < 4:
        return 0.0, 0.0

    x1, y1, x2, y2 = map(safe_float, bbox[:4])

    return (
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def intersection_area(
    a: List[float],
    b: List[float],
) -> float:

    if len(a) < 4 or len(b) < 4:
        return 0.0

    ax1, ay1, ax2, ay2 = map(safe_float, a[:4])
    bx1, by1, bx2, by2 = map(safe_float, b[:4])

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def iou(
    a: List[float],
    b: List[float],
) -> float:

    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def overlap_ratio(
    small: List[float],
    large: List[float],
) -> float:

    area = bbox_area(small)

    if area <= 0:
        return 0.0

    return intersection_area(small, large) / area


def center_distance(
    a: List[float],
    b: List[float],
) -> float:

    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)

    return math.sqrt(
        (ax - bx) ** 2 +
        (ay - by) ** 2
    )


# ============================================================
# Text semantic helpers
# ============================================================

def looks_like_email(text: str) -> bool:

    text = normalize_text(text).lower()

    if "email" in text:
        return True

    if "username" in text:
        return True

    if "mobile number" in text:
        return True

    if "phone number" in text:
        return True

    return bool(
        re.search(
            r"\b[\w.+-]+@[\w.-]+\.\w+\b",
            text
        )
    )


def looks_like_password(text: str) -> bool:

    text = normalize_text(text).lower()

    password_words = [
        "password",
        "passcode",
        "pwd",
        "pin",
    ]

    return any(
        word in text
        for word in password_words
    )


def looks_like_button_text(text: str) -> bool:

    text = normalize_text(text).lower()

    if not text:
        return False

    button_words = [
        "log in",
        "login",
        "sign in",
        "sign up",
        "submit",
        "continue",
        "next",
        "previous",
        "save",
        "cancel",
        "delete",
        "confirm",
        "register",
        "create account",
        "create new account",
        "get started",
        "search",
        "send",
        "apply",
        "upload",
        "download",
        "checkout",
        "buy",
        "add",
        "learn more",
    ]

    return any(
        word in text
        for word in button_words
    )


def looks_like_link_text(text: str) -> bool:

    text = normalize_text(text).lower()

    if not text:
        return False

    link_words = [
        "forgot password",
        "privacy",
        "terms",
        "help",
        "learn more",
        "contact",
        "about",
        "cookies",
        "instagram",
        "facebook",
        "twitter",
        "linkedin",
        "github",
    ]

    return any(
        word in text
        for word in link_words
    )


def looks_like_label(text: str) -> bool:

    text = normalize_text(text).lower()

    if not text:
        return False

    if looks_like_email(text):
        return True

    if looks_like_password(text):
        return True

    label_words = [
        "name",
        "username",
        "email",
        "phone",
        "mobile",
        "address",
        "search",
        "date",
        "first name",
        "last name",
    ]

    return any(
        word in text
        for word in label_words
    )


# ============================================================
# Prepare YOLO results
# ============================================================

def prepare_yolo(
    detections: Any,
    min_confidence: float = 0.02,
) -> List[Dict[str, Any]]:

    prepared = []

    if detections is None:
        return prepared

    # ScreenParser returns:
    # {"detections": [...]}

    if isinstance(detections, dict):

        if "detections" in detections:
            detections = detections["detections"]

        elif "elements" in detections:
            detections = detections["elements"]

        else:
            detections = [detections]

    if not isinstance(detections, (list, tuple)):
        return prepared

    for item in detections:

        if not isinstance(item, dict):
            continue

        item = deepcopy(item)

        confidence = safe_float(
            item.get("confidence", 0.0)
        )

        if confidence < min_confidence:
            continue

        bbox = item.get("bbox")

        if not isinstance(bbox, (list, tuple)):
            continue

        if len(bbox) < 4:
            continue

        bbox = [
            safe_float(v)
            for v in bbox[:4]
        ]

        width, height = bbox_size(bbox)

        if width <= 0 or height <= 0:
            continue

        item["bbox"] = bbox
        item["width"] = width
        item["height"] = height
        item["area"] = width * height
        item["center"] = list(
            bbox_center(bbox)
        )

        item["class"] = str(
            item.get("class", "Unknown")
        )

        prepared.append(item)

    return prepared


# ============================================================
# Prepare OCR results
# ============================================================

def prepare_ocr(
    ocr_results: Any,
    min_confidence: float = 0.30,
) -> List[Dict[str, Any]]:

    prepared = []

    if ocr_results is None:
        return prepared

    # OCRParser returns:
    # {"texts": [...]}

    if isinstance(ocr_results, dict):

        if "texts" in ocr_results:
            ocr_results = ocr_results["texts"]

        elif "ocr" in ocr_results:
            ocr_results = ocr_results["ocr"]

        elif "results" in ocr_results:
            ocr_results = ocr_results["results"]

        else:
            ocr_results = [ocr_results]

    if not isinstance(ocr_results, (list, tuple)):
        return prepared

    for item in ocr_results:

        if not isinstance(item, dict):
            continue

        text = normalize_text(
            item.get("text", "")
        )

        confidence = safe_float(
            item.get("confidence", 0.0)
        )

        bbox = item.get("bbox")

        if not text:
            continue

        if confidence < min_confidence:
            continue

        if not isinstance(bbox, (list, tuple)):
            continue

        if len(bbox) < 4:
            continue

        bbox = [
            safe_float(v)
            for v in bbox[:4]
        ]

        width, height = bbox_size(bbox)

        if width <= 0 or height <= 0:
            continue

        prepared.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox,
            "width": width,
            "height": height,
            "area": width * height,
            "center": list(
                bbox_center(bbox)
            ),
        })

    return prepared


# ============================================================
# Element type helpers
# ============================================================

def is_input(element: Dict[str, Any]) -> bool:

    cls = str(
        element.get("class", "")
    ).lower()

    return cls in {
        "text input",
        "input",
        "textbox",
        "textinput",
    }


def is_button(element: Dict[str, Any]) -> bool:

    cls = str(
        element.get("class", "")
    ).lower()

    return cls in {
        "button",
        "utility button",
        "submit button",
        "icon button",
    }


def is_link(element: Dict[str, Any]) -> bool:

    cls = str(
        element.get("class", "")
    ).lower()

    return cls in {
        "link",
        "hyperlink",
    }


def is_text(element: Dict[str, Any]) -> bool:

    return (
        str(
            element.get("class", "")
        ).lower()
        == "text"
    )


# ============================================================
# OCR matching
# ============================================================

def score_input_match(
    ocr: Dict[str, Any],
    element: Dict[str, Any],
) -> float:

    obox = ocr["bbox"]
    ebox = element["bbox"]
    text = ocr["text"]

    score = 0.0

    overlap = overlap_ratio(
        obox,
        ebox
    )

    if overlap > 0.50:
        score += 0.55

    elif overlap > 0.20:
        score += 0.35

    elif overlap > 0.05:
        score += 0.15

    ox, _ = bbox_center(obox)
    ex, _ = bbox_center(ebox)

    _, oh = bbox_size(obox)
    _, eh = bbox_size(ebox)

    vertical_gap = ebox[1] - obox[3]

    if (
        vertical_gap >= -oh * 0.50
        and vertical_gap <= max(
            80.0,
            eh * 2.5
        )
        and abs(ox - ex) <= max(
            150.0,
            ebox[2] - ebox[0]
        )
    ):
        score += 0.35

    if overlap > 0.20:
        score += 0.25

    if abs(ox - ex) <= max(
        100.0,
        (ebox[2] - ebox[0]) * 0.50
    ):
        score += 0.10

    if looks_like_password(text):
        score += 0.25

    elif looks_like_email(text):
        score += 0.20

    elif looks_like_label(text):
        score += 0.10

    return min(score, 1.0)


def score_button_match(
    ocr: Dict[str, Any],
    element: Dict[str, Any],
) -> float:

    obox = ocr["bbox"]
    ebox = element["bbox"]
    text = ocr["text"]

    score = 0.0

    overlap = overlap_ratio(
        obox,
        ebox
    )

    if overlap > 0.50:
        score += 0.65

    elif overlap > 0.20:
        score += 0.45

    elif overlap > 0.05:
        score += 0.20

    distance = center_distance(
        obox,
        ebox
    )

    ew, eh = bbox_size(ebox)

    tolerance = max(
        30.0,
        min(
            100.0,
            max(ew, eh) * 1.5
        )
    )

    if distance <= tolerance:
        score += 0.20

    if looks_like_button_text(text):
        score += 0.25

    return min(score, 1.0)


def score_link_match(
    ocr: Dict[str, Any],
    element: Dict[str, Any],
) -> float:

    obox = ocr["bbox"]
    ebox = element["bbox"]
    text = ocr["text"]

    score = 0.0

    overlap = overlap_ratio(
        obox,
        ebox
    )

    if overlap > 0.50:
        score += 0.65

    elif overlap > 0.20:
        score += 0.45

    elif overlap > 0.05:
        score += 0.20

    distance = center_distance(
        obox,
        ebox
    )

    ew, eh = bbox_size(ebox)

    tolerance = max(
        30.0,
        min(
            80.0,
            max(ew, eh) * 1.5
        )
    )

    if distance <= tolerance:
        score += 0.20

    if looks_like_link_text(text):
        score += 0.20

    return min(score, 1.0)


def score_text_match(
    ocr: Dict[str, Any],
    element: Dict[str, Any],
) -> float:

    obox = ocr["bbox"]
    ebox = element["bbox"]

    overlap = overlap_ratio(
        obox,
        ebox
    )

    if overlap > 0.50:
        return 0.90

    if overlap > 0.20:
        return 0.60

    if overlap > 0.05:
        return 0.35

    distance = center_distance(
        obox,
        ebox
    )

    ew, eh = bbox_size(ebox)

    tolerance = max(
        15.0,
        min(
            50.0,
            max(ew, eh) * 0.5
        )
    )

    if distance <= tolerance:
        return 0.30

    return 0.0


# ============================================================
# Input classification
# ============================================================

def classify_input_type(
    element: Dict[str, Any],
) -> str:

    texts = []

    for key in [
        "text",
        "label",
        "placeholder",
    ]:

        value = normalize_text(
            element.get(key)
        )

        if value:
            texts.append(value)

    combined = " ".join(texts)

    if looks_like_password(combined):
        return "password"

    if looks_like_email(combined):
        return "email"

    return "text"


# ============================================================
# Attach OCR
# ============================================================

def attach_to_input(
    element: Dict[str, Any],
    ocr: Dict[str, Any],
    score: float,
) -> None:

    text = ocr["text"]
    confidence = ocr["confidence"]

    if looks_like_password(text):

        element["placeholder"] = text
        element["placeholder_confidence"] = confidence
        element["placeholder_match_score"] = score
        element["input_type"] = "password"

        return

    if looks_like_email(text):

        element["label"] = text
        element["label_confidence"] = confidence
        element["label_match_score"] = score
        element["input_type"] = "email"

        return

    if looks_like_label(text):

        element["label"] = text
        element["label_confidence"] = confidence
        element["label_match_score"] = score
        element["input_type"] = classify_input_type(element)

        return

    if overlap_ratio(
        ocr["bbox"],
        element["bbox"]
    ) > 0.20:

        element["text"] = text
        element["text_confidence"] = confidence
        element["text_match_score"] = score


def attach_to_button(
    element: Dict[str, Any],
    ocr: Dict[str, Any],
    score: float,
) -> None:

    element["text"] = ocr["text"]
    element["text_confidence"] = ocr["confidence"]
    element["text_match_score"] = score


def attach_to_link(
    element: Dict[str, Any],
    ocr: Dict[str, Any],
    score: float,
) -> None:

    element["text"] = ocr["text"]
    element["text_confidence"] = ocr["confidence"]
    element["text_match_score"] = score


def attach_to_text(
    element: Dict[str, Any],
    ocr: Dict[str, Any],
    score: float,
) -> None:

    element["text"] = ocr["text"]
    element["text_confidence"] = ocr["confidence"]
    element["text_match_score"] = score


# ============================================================
# Candidate matching
# ============================================================

def collect_candidates(
    ocr_list: List[Dict[str, Any]],
    elements: List[Dict[str, Any]],
) -> List[Tuple[int, int, float]]:

    candidates = []

    for ocr_index, ocr_item in enumerate(ocr_list):

        for element_index, element in enumerate(elements):

            if is_input(element):

                score = score_input_match(
                    ocr_item,
                    element
                )

                threshold = 0.30

            elif is_button(element):

                score = score_button_match(
                    ocr_item,
                    element
                )

                threshold = 0.35

            elif is_link(element):

                score = score_link_match(
                    ocr_item,
                    element
                )

                threshold = 0.35

            elif is_text(element):

                score = score_text_match(
                    ocr_item,
                    element
                )

                threshold = 0.30

            else:
                continue

            if score >= threshold:

                candidates.append(
                    (
                        ocr_index,
                        element_index,
                        score
                    )
                )

    candidates.sort(
        key=lambda item: item[2],
        reverse=True
    )

    return candidates


# ============================================================
# Duplicate handling
# ============================================================

def _merge_ocr_fields(
    survivor: Dict[str, Any],
    loser: Dict[str, Any],
) -> None:

    transferable_keys = (
        "text",
        "text_confidence",
        "text_match_score",
        "label",
        "label_confidence",
        "label_match_score",
        "placeholder",
        "placeholder_confidence",
        "placeholder_match_score",
        "input_type",
    )

    for key in transferable_keys:

        if (
            key in loser
            and key not in survivor
        ):
            survivor[key] = loser[key]


def remove_semantic_duplicates(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    result = []

    structural = [
        element
        for element in elements
        if (
            is_input(element)
            or is_button(element)
            or is_link(element)
        )
    ]

    for element in elements:

        if not is_text(element):

            result.append(element)
            continue

        text = normalize_text(
            element.get("text", "")
        )

        if not text:

            result.append(element)
            continue

        remove = False

        for target in structural:

            overlap = overlap_ratio(
                element["bbox"],
                target["bbox"]
            )

            if overlap >= 0.35:

                remove = True
                break

        if not remove:
            result.append(element)

    return result


def remove_duplicate_elements(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not elements:
        return []

    priority = {
        "Text Input": 100,
        "Button": 95,
        "Link": 90,
        "Image": 80,
        "Select": 75,
        "Checkbox": 70,
        "Text": 30,
    }

    elements = sorted(
        elements,
        key=lambda element: (
            priority.get(
                str(element.get("class")),
                10
            ),
            1 if (
                element.get("text")
                or element.get("label")
                or element.get("placeholder")
            ) else 0,
            safe_float(
                element.get("confidence", 0.0)
            ),
        ),
        reverse=True,
    )

    kept = []

    for current in elements:

        duplicate_of = None

        for existing in kept:

            same_class = (
                current.get("class")
                ==
                existing.get("class")
            )

            if (
                is_text(current)
                and (
                    is_input(existing)
                    or is_button(existing)
                    or is_link(existing)
                )
            ):

                if overlap_ratio(
                    current["bbox"],
                    existing["bbox"]
                ) >= 0.30:

                    duplicate_of = existing
                    break

            if not same_class:
                continue

            overlap = iou(
                current["bbox"],
                existing["bbox"]
            )

            current_area = bbox_area(
                current["bbox"]
            )

            existing_area = bbox_area(
                existing["bbox"]
            )

            if current_area <= 0:
                continue

            smaller = min(
                current_area,
                existing_area
            )

            intersection = intersection_area(
                current["bbox"],
                existing["bbox"]
            )

            contained_ratio = (
                intersection / smaller
                if smaller > 0
                else 0
            )

            if (
                overlap >= 0.70
                or contained_ratio >= 0.80
            ):

                duplicate_of = existing
                break

        if duplicate_of is not None:

            _merge_ocr_fields(
                duplicate_of,
                current
            )

            continue

        kept.append(current)

    return kept


# ============================================================
# Main merger
# ============================================================

class UIMerger:

    def __init__(
        self,
        min_yolo_confidence: float = 0.02,
        min_ocr_confidence: float = 0.30,
        min_match_score: float = 0.30,
        remove_duplicates: bool = True,
    ):

        self.min_yolo_confidence = min_yolo_confidence
        self.min_ocr_confidence = min_ocr_confidence
        self.min_match_score = min_match_score
        self.remove_duplicates = remove_duplicates

    def merge(
        self,
        yolo_detections: Any,
        ocr_results: Any,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Prepare YOLO
        # ----------------------------------------------------

        yolo = prepare_yolo(
            yolo_detections,
            self.min_yolo_confidence
        )

        # ----------------------------------------------------
        # Prepare OCR
        # ----------------------------------------------------

        ocr = prepare_ocr(
            ocr_results,
            self.min_ocr_confidence
        )

        elements = deepcopy(yolo)

        matched_ocr = set()
        claimed_elements = set()

        # ----------------------------------------------------
        # Global OCR -> UI matching
        # ----------------------------------------------------

        candidates = collect_candidates(
            ocr,
            elements
        )

        for (
            ocr_index,
            element_index,
            score
        ) in candidates:

            if ocr_index in matched_ocr:
                continue

            if element_index in claimed_elements:
                continue

            element = elements[element_index]
            ocr_item = ocr[ocr_index]

            if is_input(element):

                attach_to_input(
                    element,
                    ocr_item,
                    score
                )

            elif is_button(element):

                attach_to_button(
                    element,
                    ocr_item,
                    score
                )

            elif is_link(element):

                attach_to_link(
                    element,
                    ocr_item,
                    score
                )

            elif is_text(element):

                attach_to_text(
                    element,
                    ocr_item,
                    score
                )

            else:
                continue

            matched_ocr.add(ocr_index)
            claimed_elements.add(element_index)

        # ----------------------------------------------------
        # Add genuine standalone OCR text
        # ----------------------------------------------------

        for ocr_index, ocr_item in enumerate(ocr):

            if ocr_index in matched_ocr:
                continue

            text = normalize_text(
                ocr_item.get("text", "")
            )

            if not text:
                continue

            represented = False

            for element in elements:

                if not (
                    is_input(element)
                    or is_button(element)
                    or is_link(element)
                ):
                    continue

                overlap = overlap_ratio(
                    ocr_item["bbox"],
                    element["bbox"]
                )

                if overlap >= 0.30:

                    represented = True
                    break

                distance = center_distance(
                    ocr_item["bbox"],
                    element["bbox"]
                )

                ew, eh = bbox_size(
                    element["bbox"]
                )

                tolerance = max(
                    25.0,
                    min(
                        80.0,
                        max(ew, eh) * 0.75
                    )
                )

                if distance <= tolerance:

                    if (
                        is_input(element)
                        and (
                            looks_like_label(text)
                            or looks_like_password(text)
                        )
                    ):
                        represented = True
                        break

                    if (
                        is_button(element)
                        and looks_like_button_text(text)
                    ):
                        represented = True
                        break

                    if (
                        is_link(element)
                        and looks_like_link_text(text)
                    ):
                        represented = True
                        break

            if represented:
                continue

            elements.append({
                "class": "Text",
                "confidence": ocr_item["confidence"],
                "bbox": ocr_item["bbox"],
                "width": ocr_item["width"],
                "height": ocr_item["height"],
                "area": ocr_item["area"],
                "center": ocr_item["center"],
                "text": text,
                "text_confidence": ocr_item["confidence"],
                "text_match_score": 0.0,
            })

        # ----------------------------------------------------
        # Finalize input types
        # ----------------------------------------------------

        for element in elements:

            if is_input(element):

                element["input_type"] = (
                    classify_input_type(element)
                )

        # ----------------------------------------------------
        # Remove semantic duplicates
        # ----------------------------------------------------

        elements = remove_semantic_duplicates(
            elements
        )

        # ----------------------------------------------------
        # Remove geometric duplicates
        # ----------------------------------------------------

        if self.remove_duplicates:

            elements = remove_duplicate_elements(
                elements
            )

        # ----------------------------------------------------
        # Reading order
        # ----------------------------------------------------

        elements.sort(
            key=lambda element: (
                safe_float(
                    element["bbox"][1]
                ),
                safe_float(
                    element["bbox"][0]
                ),
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        statistics = {
            "yolo_input": (
                len(yolo_detections.get("detections", []))
                if isinstance(yolo_detections, dict)
                else len(yolo_detections)
                if isinstance(yolo_detections, list)
                else len(yolo)
            ),
            "yolo_after_filter": len(yolo),

            "ocr_input": (
                len(ocr_results.get("texts", []))
                if isinstance(ocr_results, dict)
                else len(ocr_results)
                if isinstance(ocr_results, list)
                else len(ocr)
            ),
            "ocr_valid": len(ocr),

            "final_elements": len(elements),
            "ocr_matched": len(matched_ocr),
            "ocr_unmatched": len(ocr) - len(matched_ocr),
        }

        return {
            "elements": elements,
            "statistics": statistics,
        }


# ============================================================
# Convenience function
# ============================================================

def merge_ui(
    yolo_results: Any,
    ocr_results: Any,
) -> Dict[str, Any]:

    merger = UIMerger(
        min_yolo_confidence=0.02,
        min_ocr_confidence=0.30,
        min_match_score=0.30,
        remove_duplicates=True,
    )

    return merger.merge(
        yolo_results,
        ocr_results,
    )
