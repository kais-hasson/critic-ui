import os

from google import genai


def create_gemini_client():
    """
    Create and return a Gemini API client.

    The API key is read from the GEMINI_API_KEY
    environment variable.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY environment variable "
            "is not set."
        )

    return genai.Client(
        api_key=api_key
    )