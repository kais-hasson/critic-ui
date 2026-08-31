"""
Main CriticUI Pipeline

Screenshot
    ↓
ScreenParser
    ↓
OCR
    ↓
Merger
    ↓
OpenCV Feature Extraction
    ↓
Gemini LLM
    ↓
Affected Elements Expansion
    ↓
Final UI/UX Analysis
"""

from pprint import pprint

from app.config import create_gemini_client

from app.models.ocr import OCRParser
from app.models.screen_parser import ScreenParser

from app.services.merger import UIMerger

from app.services.opencv.feature_extractor import (
    extract_ui_features,
    merge_with_detection_data,
)

from app.services.llm.gemini_analyzer import (
    UIUXGeminiAnalyzer,
)

from app.services.llm.expand_affected_elements import (
    AffectedElementsExpander,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_PATH = "uploads/Screenshot.png"

MODEL_PATH = "weights/best.pt"

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# Main Pipeline
# ============================================================

def analyze_image(
    image_path: str = IMAGE_PATH,
    model_path: str = MODEL_PATH,
):
    """
    Run the complete CriticUI analysis pipeline.
    """

    print("=" * 60)
    print("CRITIC UI ANALYSIS")
    print("=" * 60)

    # ========================================================
    # 1. ScreenParser
    # ========================================================

    print("\n[1/6] Starting ScreenParser...")

    screen_parser = ScreenParser(
        model_path=model_path
    )

    parser_result = screen_parser.parse(
        image_path=image_path,
        min_confidence=0.01,
    )

    print(
        f"Detected elements: "
        f"{len(parser_result.get('detections', []))}"
    )

    # ========================================================
    # 2. OCR
    # ========================================================

    print("\n[2/6] Starting OCR...")

    ocr = OCRParser()

    ocr_result = ocr.parse(
        image_path
    )

    print(
        f"Detected texts: "
        f"{len(ocr_result.get('texts', []))}"
    )

    # ========================================================
    # 3. Merger
    # ========================================================

    print("\n[3/6] Starting Merger...")

    merger = UIMerger()

    merger_result = merger.merge(
        parser_result.get(
            "detections",
            []
        ),
        ocr_result.get(
            "texts",
            []
        ),
    )

    print(
        f"Merged elements: "
        f"{len(merger_result.get('elements', []))}"
    )

    # ========================================================
    # 4. OpenCV Feature Extraction
    # ========================================================

    print(
        "\n[4/6] Starting OpenCV "
        "Feature Extraction..."
    )

    features = extract_ui_features(
        image_path=image_path,
        merger_output=merger_result,
    )

    # ========================================================
    # Build Final Structured Input
    # ========================================================

    final_result = merge_with_detection_data(
        merger_result,
        features,
    )

    print("\nOpenCV analysis completed.")

    # ========================================================
    # 5. Gemini LLM
    # ========================================================

    print("\n[5/6] Starting Gemini UI/UX Analysis...")

    client = create_gemini_client()

    analyzer = UIUXGeminiAnalyzer(
        client=client,
        model=GEMINI_MODEL,
    )

    analysis = analyzer.analyze(
        final_result
    )

    # --------------------------------------------------------
    # Check Gemini error
    # --------------------------------------------------------

    if (
        isinstance(analysis, dict)
        and "error" in analysis
    ):

        print(
            "\nGemini analysis failed:"
        )

        pprint(
            analysis,
            sort_dicts=False,
        )

        return {
            "final_result": final_result,
            "analysis": analysis,
        }

    print("\nGemini analysis completed.")

    # ========================================================
    # 6. Expand affected_elements
    # ========================================================

    print(
        "\n[6/6] Expanding affected elements..."
    )

    expander = AffectedElementsExpander()

    final_analysis = expander.expand(
        analysis=analysis,
        final_result=final_result,
    )

    print(
        "\nAffected elements expansion completed."
    )

    # ========================================================
    # Final Output
    # ========================================================

    result = {
        "ui_data": final_result,
        "analysis": final_analysis,
    }

    print("\n" + "=" * 60)
    print("FINAL UI/UX ANALYSIS")
    print("=" * 60)

    pprint(
        result,
        sort_dicts=False,
    )

    return result


# ============================================================
# Local Execution
# ============================================================

if __name__ == "__main__":

    analyze_image()