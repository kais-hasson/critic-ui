import json
import re
from typing import Any, Dict


class AffectedElementsExpander:

    # ========================================================
    # 1. Build Element Lookup
    # ========================================================

    @staticmethod
    def build_element_lookup(
        final_result: Dict[str, Any],
    ) -> Dict[int, Dict[str, Any]]:

        elements = final_result.get(
            "elements",
            [],
        )

        if not isinstance(elements, list):
            elements = []

        element_lookup = {}

        for index, element in enumerate(elements):

            if not isinstance(element, dict):
                continue

            full_element = dict(element)

            full_element["id"] = index

            element_lookup[index] = full_element

        return element_lookup

    # ========================================================
    # 2. Extract IDs From Evidence
    # ========================================================

    @staticmethod
    def extract_ids_from_evidence(
        evidence,
        valid_ids,
    ):

        if not isinstance(evidence, str):
            return []

        found_ids = []

        # ----------------------------------------------------
        # Pattern 1
        #
        # Element 77
        # element 77
        # ELEMENT 77
        # ----------------------------------------------------

        matches = re.findall(
            r"\b[Ee]lement\s+(\d+)\b",
            evidence,
        )

        for value in matches:

            try:
                element_id = int(value)
            except ValueError:
                continue

            if (
                element_id in valid_ids
                and element_id not in found_ids
            ):
                found_ids.append(element_id)

        # ----------------------------------------------------
        # Pattern 2
        #
        # ID 92
        # IDs 92, 94
        # IDs 92 and 94
        # ----------------------------------------------------

        matches = re.findall(
            r"\bIDs?\s+((?:\d+(?:\s*,\s*|\s+and\s+|\s+)\s*)+\d+|\d+)",
            evidence,
            flags=re.IGNORECASE,
        )

        for group in matches:

            numbers = re.findall(
                r"\d+",
                group,
            )

            for value in numbers:

                try:
                    element_id = int(value)
                except ValueError:
                    continue

                if (
                    element_id in valid_ids
                    and element_id not in found_ids
                ):
                    found_ids.append(element_id)

        # ----------------------------------------------------
        # Pattern 3
        #
        # ID: 95
        # ID #95
        # ID 95
        # ----------------------------------------------------

        matches = re.findall(
            r"\bID\s*[:#]?\s*(\d+)\b",
            evidence,
            flags=re.IGNORECASE,
        )

        for value in matches:

            try:
                element_id = int(value)
            except ValueError:
                continue

            if (
                element_id in valid_ids
                and element_id not in found_ids
            ):
                found_ids.append(element_id)

        return found_ids

    # ========================================================
    # 3. Expand Analysis
    # ========================================================

    def expand(
        self,
        analysis: Dict[str, Any],
        final_result: Dict[str, Any],
    ):

        if not isinstance(analysis, dict):
            raise TypeError(
                "analysis must be a dictionary."
            )

        if not isinstance(final_result, dict):
            raise TypeError(
                "final_result must be a dictionary."
            )

        element_lookup = self.build_element_lookup(
            final_result
        )

        valid_element_ids = set(
            element_lookup.keys()
        )

        issues = analysis.get(
            "issues",
            [],
        )

        if not isinstance(issues, list):
            issues = []

        for issue in issues:

            if not isinstance(issue, dict):
                continue

            # ------------------------------------------------
            # Existing affected_elements
            # ------------------------------------------------

            affected_ids = issue.get(
                "affected_elements",
                [],
            )

            if not isinstance(
                affected_ids,
                list,
            ):
                affected_ids = []

            valid_ids = []

            # ------------------------------------------------
            # First use IDs returned by Gemini
            # ------------------------------------------------

            for element_id in affected_ids:

                try:
                    element_id = int(element_id)
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    element_id in element_lookup
                    and element_id not in valid_ids
                ):
                    valid_ids.append(element_id)

            # ------------------------------------------------
            # If Gemini returned no valid IDs,
            # recover IDs from evidence.
            # ------------------------------------------------

            if not valid_ids:

                evidence = issue.get(
                    "evidence",
                    "",
                )

                recovered_ids = (
                    self.extract_ids_from_evidence(
                        evidence,
                        valid_element_ids,
                    )
                )

                if recovered_ids:

                    print(
                        f"Issue {issue.get('id')}: "
                        f"Recovered IDs from evidence: "
                        f"{recovered_ids}"
                    )

                    valid_ids.extend(
                        recovered_ids
                    )

            # ------------------------------------------------
            # Convert IDs to complete elements
            # ------------------------------------------------

            expanded_elements = []

            for element_id in valid_ids:

                element = element_lookup.get(
                    element_id
                )

                if element is None:
                    continue

                expanded_elements.append(
                    dict(element)
                )

            # ------------------------------------------------
            # Replace IDs with complete objects
            # ------------------------------------------------

            issue["affected_elements"] = (
                expanded_elements
            )

        return analysis