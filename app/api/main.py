from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.main import analyze_image


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="CriticUI API",
    description="AI-powered UI/UX screenshot analysis API",
    version="1.0.0",
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "CriticUI API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


# ============================================================
# Analyze Screenshot
# ============================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Allowed: PNG, JPG, JPEG, WEBP."
            ),
        )

    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    filename = (
        f"{uuid4().hex}"
        f"{extension}"
    )

    image_path = UPLOAD_DIR / filename

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    try:

        with open(
            image_path,
            "wb",
        ) as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                buffer.write(chunk)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save image: {str(e)}"
            ),
        )

    # --------------------------------------------------------
    # Run CriticUI Pipeline
    # --------------------------------------------------------

    try:

        result = analyze_image(
            image_path=str(image_path),
        )

        return {
            "success": True,
            "filename": filename,
            "result": result,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Analysis failed.",
                "error": str(e),
            },
        )