import json

from app.models.llm_models import UIUXAnalysis


class UIUXGeminiAnalyzer:

    def __init__(
        self,
        client,
        model="gemini-3.6-flash",
    ):
        self.client = client
        self.model = model

    # ========================================================
    # 1. Prepare LLM Input
    # ========================================================

    def prepare_llm_input(self, result):

        if not isinstance(result, dict):
            raise TypeError(
                "result must be a dictionary"
            )

        elements = result.get(
            "elements",
            []
        )

        features = result.get(
            "features",
            {}
        )

        if not isinstance(elements, list):
            elements = []

        if not isinstance(features, dict):
            features = {}

        simplified_elements = []

        # ====================================================
        # Elements
        # ====================================================

        for index, element in enumerate(elements):

            if not isinstance(element, dict):
                continue

            item = {
                "id": index,
                "class": element.get("class"),
                "bbox": element.get("bbox"),
                "width": element.get("width"),
                "height": element.get("height"),
            }

            if element.get("text"):
                item["text"] = element.get("text")

            if element.get("input_type"):
                item["input_type"] = element.get(
                    "input_type"
                )

            if element.get("placeholder"):
                item["placeholder"] = element.get(
                    "placeholder"
                )

            if element.get("label"):
                item["label"] = element.get(
                    "label"
                )

            simplified_elements.append(item)

        # ====================================================
        # Features
        # ====================================================

        spacing = features.get("spacing", {})
        alignment = features.get("alignment", {})
        density = features.get("density", {})
        contrast = features.get("contrast", {})
        screen = features.get("screen", {})

        if not isinstance(spacing, dict):
            spacing = {}

        if not isinstance(alignment, dict):
            alignment = {}

        if not isinstance(density, dict):
            density = {}

        if not isinstance(contrast, dict):
            contrast = {}

        if not isinstance(screen, dict):
            screen = {}

        # ====================================================
        # Final LLM Input
        # ====================================================

        return {
            "screen": {
                "width": screen.get("width"),
                "height": screen.get("height"),
            },

            "element_count": len(
                simplified_elements
            ),

            "elements": simplified_elements,

            "metrics": {

                "spacing": {
                    "average": spacing.get("average"),
                    "minimum": spacing.get("minimum"),
                    "maximum": spacing.get("maximum"),
                    "std": spacing.get("std"),
                    "consistency": spacing.get(
                        "consistency"
                    ),
                },

                "alignment": {
                    "score": alignment.get("score"),
                    "horizontal_score": alignment.get(
                        "horizontal_score"
                    ),
                    "vertical_score": alignment.get(
                        "vertical_score"
                    ),
                },

                "density": {
                    "density": density.get("density"),
                    "empty_ratio": density.get(
                        "empty_ratio"
                    ),
                    "element_count": density.get(
                        "element_count"
                    ),
                },

                "contrast": {
                    "average_ratio": contrast.get(
                        "average_ratio"
                    ),
                },
            },
        }

    # ========================================================
    # 2. Create Prompt
    # ========================================================

    def create_prompt(self, llm_input):

        ui_data = json.dumps(
            llm_input,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return f"""
You are an expert UI/UX design evaluator.

You are analyzing a user interface using structured
information extracted from a screenshot.

The data was produced by:

- UI element detection
- OCR
- UI element merging
- Geometry analysis
- Spacing analysis
- Alignment analysis
- Density analysis
- Contrast analysis

Your job is to identify REAL and MEANINGFUL UI/UX problems.

IMPORTANT RULES:

1. Use ONLY the provided UI data.

2. Do NOT invent information.

3. Do NOT assume visual properties that are not present
   in the provided data.

4. Every issue MUST have evidence.

5. Every strength MUST have evidence.

6. Do NOT report a problem only because a single metric
   is slightly unusual.

7. Ignore obvious detection noise.

8. Do NOT create duplicate issues.

9. Use the actual element IDs from the input.

10. affected_elements must contain only valid element IDs.

11. Focus on meaningful problems related to:

   - spacing
   - alignment
   - visual hierarchy
   - density
   - contrast
   - layout consistency
   - form usability
   - accessibility

12. If there is not enough evidence to prove a problem,
    do NOT report it.

13. Recommendations must directly address the reported issue.

14. Severity must be:

   low
   medium
   high

15. overall_score must be between 0 and 100.

16. If there are no reliable issues, return an empty issues array.

17. Do not invent strengths.

18. Return ONLY the requested structured output.

UI DATA:

{ui_data}
"""

    # ========================================================
    # 3. Analyze
    # ========================================================

    def analyze(self, result):

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        llm_input = self.prepare_llm_input(
            result
        )

        # ----------------------------------------------------
        # Create prompt
        # ----------------------------------------------------

        prompt = self.create_prompt(
            llm_input
        )

        # ----------------------------------------------------
        # Structured Output Schema
        # ----------------------------------------------------

        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": UIUXAnalysis.model_json_schema(),
        }

        # ====================================================
        # Gemini Request
        # ====================================================

        try:

            interaction = self.client.interactions.create(
                model=self.model,
                input=prompt,
                response_format=response_format,
            )

        except Exception as e:

            return {
                "error": "Gemini API request failed",
                "details": str(e),
            }

        # ====================================================
        # Get output
        # ====================================================

        output_text = getattr(
            interaction,
            "output_text",
            None,
        )

        if not output_text:

            return {
                "error": "Gemini returned empty output",
                "raw_response": str(interaction),
            }

        # ====================================================
        # Validate Pydantic output
        # ====================================================

        try:

            analysis = UIUXAnalysis.model_validate_json(
                output_text
            )

        except Exception as e:

            return {
                "error": "Failed to validate Gemini output",
                "details": str(e),
                "raw_response": output_text,
            }

        # ====================================================
        # Normalize score
        # ====================================================

        score = max(
            0,
            min(
                100,
                analysis.overall_score,
            ),
        )

        analysis.overall_score = score

        # ====================================================
        # Return dictionary
        # ====================================================

        return analysis.model_dump()