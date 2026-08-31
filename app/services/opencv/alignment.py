"""
UI alignment analysis.

--------------------------------------------------------------
FIX (2026-08-26): calculate_axis_score() rewritten.
--------------------------------------------------------------
PROBLEM:
The original implementation compared EVERY pair of elements on
the entire screen against each other (a full O(n^2) all-pairs
scan), then reported:

    aligned_pairs / total_pairs

With ~129 elements, that is C(129, 2) = 8,256 pairs. Only pairs
that belong to the SAME logical row/column (e.g. two cells in
the same table row, or two headers in the same column) can ever
be "aligned". Pairs that have no logical relationship at all
(e.g. a header element near the top of the screen vs. a table
cell near the bottom) will essentially never satisfy the
tolerance check, yet they are still counted as "unaligned pairs"
in the denominator.

Because the vast majority of all possible pairs on a real screen
are exactly these unrelated, distant pairs, the ratio collapses
towards zero even when the actual row/column alignment (e.g. a
neatly organized data table) is excellent. This is exactly why a
well-aligned table screenshot still produced a score of ~0.05.

FIX:
Instead of scoring "what fraction of ALL possible pairs align",
score "what fraction of elements have at least one alignment
partner". For each element, check whether there exists at least
one OTHER element whose corresponding edge/center falls within
`tolerance` of it. This:

  - No longer penalizes the score for the huge number of
    pairs between elements that were never expected to relate
    to each other in the first place.
  - Still correctly rewards a screen where rows/columns are
    consistently aligned, regardless of how many total elements
    are on screen.
  - Runtime remains O(n log n) via sorting instead of O(n^2)
    comparisons, which also scales far better on dense UIs.
"""
from __future__ import annotations

from typing import List

from .ui_types import(
    Detection,
    AlignmentAnalysis,
)


DEFAULT_MIN_CONFIDENCE = 0.30


def filter_detections(
    detections: List[Detection],
    min_confidence: float,
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


def calculate_axis_score(
    values: List[float],
    tolerance: float = 5.0,
) -> float:
    """
    FIX (2026-08-26): "at least one partner" scoring instead of
    "fraction of all possible pairs".

    For each value, we check whether at least one OTHER value in
    the list lies within `tolerance` of it. The score is the
    fraction of values that have such a partner.

    This is implemented via a sort + two-pointer sweep so it
    stays O(n log n) instead of the previous O(n^2) all-pairs
    scan, and -- more importantly -- it does not get diluted by
    the large number of unrelated far-apart pairs that a full
    all-pairs comparison inevitably includes on any real screen
    with more than a handful of elements.
    """
    n = len(values)
    if n < 2:
        return 0.0

    # Sort while keeping track of which sorted position corresponds
    # to which original element, so we can mark each one as
    # "has a partner" independently.
    indexed = sorted(
        range(n),
        key=lambda i: values[i],
    )
    sorted_values = [values[i] for i in indexed]

    has_partner = [False] * n

    left = 0
    for right in range(n):
        # Shrink the window from the left while it's too far from
        # the current right edge to still be within tolerance.
        while (
            sorted_values[right]
            - sorted_values[left]
            > tolerance
        ):
            left += 1

        # Every element within [left, right] is mutually within
        # `tolerance` of sorted_values[right] on this axis (since
        # the list is sorted, the max gap in the window is
        # sorted_values[right] - sorted_values[left]).
        if right > left:
            for k in range(left, right + 1):
                has_partner[indexed[k]] = True

    aligned_count = sum(has_partner)
    return aligned_count / n


def _group_alignment_score(
    left_edges: List[float],
    right_edges: List[float],
    centers: List[float],
    tolerance: float,
) -> float:
    return max(
        calculate_axis_score(left_edges, tolerance),
        calculate_axis_score(right_edges, tolerance),
        calculate_axis_score(centers, tolerance),
    )


def calculate_alignment_analysis(
    detections: List[Detection],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    tolerance: float = 5.0,
) -> AlignmentAnalysis:
    filtered = filter_detections(
        detections,
        min_confidence,
    )

    if len(filtered) < 2:
        return AlignmentAnalysis(
            status="insufficient_data"
        )

    left_edges = [d.bbox.x1 for d in filtered]
    right_edges = [d.bbox.x2 for d in filtered]
    horizontal_centers = [d.bbox.center[0] for d in filtered]

    top_edges = [d.bbox.y1 for d in filtered]
    bottom_edges = [d.bbox.y2 for d in filtered]
    vertical_centers = [d.bbox.center[1] for d in filtered]

    horizontal_score = _group_alignment_score(
        left_edges,
        right_edges,
        horizontal_centers,
        tolerance,
    )
    vertical_score = _group_alignment_score(
        top_edges,
        bottom_edges,
        vertical_centers,
        tolerance,
    )

    score = (
        horizontal_score
        + vertical_score
    ) / 2.0

    return AlignmentAnalysis(
        score=float(score),
        horizontal_score=float(
            horizontal_score
        ),
        vertical_score=float(
            vertical_score
        ),
        status="ok",
    )