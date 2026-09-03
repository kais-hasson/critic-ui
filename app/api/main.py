from pathlib import Path
from uuid import uuid4
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.main import analyze_image
from app.auth.dependencies import (
    get_current_user,
    get_db,
)
from app.auth.routes import router as auth_router
from app.database.user import User
from app.database.analysis import Analysis


# ============================================================
# Request Schemas
# ============================================================
class AnalysisUpdate(BaseModel):
    result: dict[str, Any]


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="CriticUI API",
    description="AI-powered UI/UX screenshot analysis API",
    version="1.0.0",
)

# ============================================================
# Authentication Routes
# ============================================================
app.include_router(auth_router)

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
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
    extension = Path(file.filename).suffix.lower()
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
    filename = f"{uuid4().hex}{extension}"
    image_path = UPLOAD_DIR / filename

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------
    try:
        with open(image_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save image: {str(e)}",
        )

    # --------------------------------------------------------
    # Run CriticUI Pipeline
    # --------------------------------------------------------
    try:
        result = analyze_image(image_path=str(image_path))

        # ----------------------------------------------------
        # Save Analysis in Database
        # ----------------------------------------------------
        analysis = Analysis(
            user_id=current_user.id,
            image_path=str(image_path),
            result=result,
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        # ----------------------------------------------------
        # Return Result
        # ----------------------------------------------------
        return {
            "success": True,
            "analysis_id": analysis.id,
            "filename": filename,
            "result": result,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Analysis failed.",
                "error": str(e),
            },
        )


# ============================================================
# Get Current User Analyses
# ============================================================
@app.get("/analyses")
def get_my_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analyses = (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    return {
        "success": True,
        "count": len(analyses),
        "analyses": [
            {
                "id": analysis.id,
                "user_id": analysis.user_id,
                "image_path": analysis.image_path,
                "result": analysis.result,
                "created_at": analysis.created_at,
            }
            for analysis in analyses
        ],
    }


# ============================================================
# Get One Analysis
# ============================================================
@app.get("/analyses/{analysis_id}")
def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
        .first()
    )
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )
    return {
        "success": True,
        "analysis": {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "image_path": analysis.image_path,
            "result": analysis.result,
            "created_at": analysis.created_at,
        },
    }


# ============================================================
# Update One Analysis
# ============================================================
@app.patch("/analyses/{analysis_id}")
def update_analysis(
    analysis_id: int,
    data: AnalysisUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
        .first()
    )
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    # --------------------------------------------------------
    # Update only the analysis result
    # --------------------------------------------------------
    analysis.result = data.result
    db.commit()
    db.refresh(analysis)

    return {
        "success": True,
        "message": "Analysis updated successfully.",
        "analysis": {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "image_path": analysis.image_path,
            "result": analysis.result,
            "created_at": analysis.created_at,
        },
    }