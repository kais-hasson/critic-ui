import os

from dotenv import load_dotenv
from google import genai


# تحميل .env من جذر المشروع
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_PATH)


def create_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(
        api_key=api_key
    )