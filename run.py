import os

from dotenv import load_dotenv


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Validate Gemini API Key
# ============================================================

if not os.getenv("GEMINI_API_KEY"):

    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Please add it to your .env file."
    )


# ============================================================
# Import application
# ============================================================

from app.main import analyze_image


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    analyze_image()
