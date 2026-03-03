# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.config import settings
from app.database import init_db, get_db, DiagnosisORM, PatientORM
from app.models import (
    PatientCreate,
    PatientUpdate,
    PatientResponse,
    ImageAnalysisRequest,
    ImageAnalysisResult,
    SymptomAnalysisRequest,
    SymptomAnalysisResult,
    MultimodalDiagnosisRequest,
    DiagnosisReport,
    DiagnosisHistoryItem,
    DiagnosisHistoryResponse,
    HealthResponse,
    ImageModality,
)
from app.patient_manager import patient_manager
from app.image_analyzer import image_analyzer
from app.symptom_analyzer import symptom_analyzer
from app.diagnostic_engine import diagnostic_engine

# ─────────────────────────── App setup ───────────────────────────────────────

# ─────────────────────────── Lifecycle ───────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    import asyncio
    from app.image_analyzer import _BiomedCLIPModel
    await init_db()
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    # Load BiomedCLIP in a thread pool but AWAIT completion so the model is
    # guaranteed to be ready before the server begins serving requests.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _BiomedCLIPModel.get)
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    root_path=settings.ROOT_PATH,
    lifespan=lifespan,
    description=(
        "REST API for the Intelligent Multimodal Medical Diagnostic System. "
        "Combines medical image analysis (X-ray, CT, MRI, dermatology, pathology) "
        "with symptom-based reasoning to generate structured diagnostic reports."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOW_METHODS.split(","),
    allow_headers=settings.CORS_ALLOW_HEADERS.split(","),
)

# ─────────────────────────── Static UI ───────────────────────────────────────

_ui_dir = Path(__file__).parent.parent / "ui"
_ui_index = _ui_dir / "index.html"


def _serve_ui():
    if _ui_index.exists():
        return FileResponse(str(_ui_index))
    return RedirectResponse("/docs")


@app.get("/", include_in_schema=False)
async def root():
    return _serve_ui()


@app.get("/ui", include_in_schema=False)
async def ui_root():
    return _serve_ui()


@app.get("/ui/", include_in_schema=False)
async def ui_root_slash():
    return _serve_ui()


# ─────────────────────────── Health ──────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System health check",
)
async def health_check():
    """Returns service health and capability flags."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        demo_mode=settings.DEMO_MODE,
        llm_available=bool(settings.LLM_ENDPOINT_URL),
        openvino_available=image_analyzer.is_openvino_available(),
        biomedclip_available=image_analyzer.is_biomedclip_available(),
        database="connected",
    )


# ─────────────────────────── Patients ────────────────────────────────────────

@app.post(
    "/patients",
    response_model=PatientResponse,
    status_code=201,
    tags=["Patients"],
    summary="Register a new patient",
)
async def create_patient(data: PatientCreate, db: AsyncSession = Depends(get_db)):
    return await patient_manager.create(db, data)


@app.get(
    "/patients",
    response_model=List[PatientResponse],
    tags=["Patients"],
    summary="List all patients",
)
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    return await patient_manager.list_all(db, skip=skip, limit=limit, search=search)


@app.get(
    "/patients/{patient_id}",
    response_model=PatientResponse,
    tags=["Patients"],
    summary="Get patient by ID",
)
async def get_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    return await patient_manager.get(db, patient_id)


@app.patch(
    "/patients/{patient_id}",
    response_model=PatientResponse,
    tags=["Patients"],
    summary="Update patient record",
)
async def update_patient(
    patient_id: int, data: PatientUpdate, db: AsyncSession = Depends(get_db)
):
    return await patient_manager.update(db, patient_id, data)


@app.delete(
    "/patients/{patient_id}",
    tags=["Patients"],
    summary="Delete patient record",
)
async def delete_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    return await patient_manager.delete(db, patient_id)


# ─────────────────────────── Image Analysis ──────────────────────────────────

@app.post(
    "/analyze/image",
    response_model=ImageAnalysisResult,
    tags=["Analysis"],
    summary="Analyse a medical image",
)
async def analyze_image(
    patient_id: int = Form(...),
    modality: ImageModality = Form(...),
    body_part: Optional[str] = Form(None),
    clinical_notes: Optional[str] = Form(None),
    model_choice: Optional[str] = Form(None, description="Classification model: biomedclip | medsiglip | medgemma"),
    image: UploadFile = File(..., description="Medical image (JPEG / PNG / DICOM-exported PNG)"),
    db: AsyncSession = Depends(get_db),
):
    # Validate patient
    await patient_manager.get(db, patient_id)

    # Check file size
    size_limit = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    content = await image.read()
    if len(content) > size_limit:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size is {settings.MAX_IMAGE_SIZE_MB} MB.",
        )

    # Save temporarily
    ext = Path(image.filename or "image.png").suffix.lower() or ".png"
    tmp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}{ext}"
    try:
        tmp_path.write_bytes(content)
        result = await image_analyzer.analyze(
            image_path=str(tmp_path),
            modality=modality,
            body_part=body_part,
            clinical_notes=clinical_notes,
            model_choice=model_choice or "biomedclip",
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result


# ─────────────────────────── Symptom Analysis ────────────────────────────────

@app.post(
    "/analyze/symptoms",
    response_model=SymptomAnalysisResult,
    tags=["Analysis"],
    summary="Analyse patient-reported symptoms",
)
async def analyze_symptoms(
    request: SymptomAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    # Validate patient
    await patient_manager.get(db, request.patient_id)
    return symptom_analyzer.analyze(request)


# ─────────────────────────── Multimodal Diagnosis ────────────────────────────

@app.post(
    "/diagnose",
    response_model=DiagnosisReport,
    status_code=201,
    tags=["Diagnosis"],
    summary="Full multimodal diagnostic report",
)
async def full_diagnosis(
    patient_id: int = Form(...),
    requesting_physician: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None, description="Comma-separated symptom list"),
    duration_days: Optional[int] = Form(None),
    clinical_notes: Optional[str] = Form(None),
    modality: Optional[ImageModality] = Form(None),
    model_choice: Optional[str] = Form(None, description="Classification model: biomedclip | medsiglip | medgemma"),
    image: Optional[UploadFile] = File(None, description="Optional medical image"),
    db: AsyncSession = Depends(get_db),
):
    # Load patient
    patient = await patient_manager.get(db, patient_id)

    # ── Image analysis ─────────────────────────────────────────────────────
    image_result: Optional[ImageAnalysisResult] = None
    if image and modality:
        content = await image.read()
        size_limit = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(content) > size_limit:
            raise HTTPException(status_code=413, detail="Image too large.")
        ext = Path(image.filename or "image.png").suffix.lower() or ".png"
        tmp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}{ext}"
        try:
            tmp_path.write_bytes(content)
            image_result = await image_analyzer.analyze(
                image_path=str(tmp_path),
                modality=modality,
                clinical_notes=clinical_notes,
                model_choice=model_choice or "biomedclip",
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Symptom analysis ───────────────────────────────────────────────────
    symptom_result: Optional[SymptomAnalysisResult] = None
    if symptoms:
        sym_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        if sym_list:
            sym_req = SymptomAnalysisRequest(
                patient_id=patient_id,
                symptoms=sym_list,
                duration_days=duration_days,
                additional_history=clinical_notes,
            )
            symptom_result = symptom_analyzer.analyze(sym_req)

    if not image_result and not symptom_result:
        raise HTTPException(
            status_code=422,
            detail="At least one of (image + modality) or symptoms must be provided.",
        )

    # ── Multimodal fusion ──────────────────────────────────────────────────
    report = await diagnostic_engine.diagnose(
        patient=patient,
        image_result=image_result,
        symptom_result=symptom_result,
        requesting_physician=requesting_physician,
    )

    # ── Persist to DB ──────────────────────────────────────────────────────
    dx_orm = DiagnosisORM(
        diagnosis_id=report.diagnosis_id,
        patient_id=patient_id,
        primary_diagnosis=report.primary_diagnosis,
        confidence=report.confidence,
        severity=report.severity.value,
        differential_diagnoses=json.dumps(report.differential_diagnoses),
        clinical_summary=report.clinical_summary,
        findings=report.findings,
        assessment=report.assessment,
        plan=report.plan,
        follow_up=report.follow_up,
        requesting_physician=requesting_physician,
        image_analysis_json=image_result.model_dump_json() if image_result else None,
        symptom_analysis_json=symptom_result.model_dump_json() if symptom_result else None,
        processing_time_ms=report.processing_time_ms,
        created_at=datetime.now(timezone.utc),
    )
    db.add(dx_orm)
    await db.commit()

    return report


# ─────────────────────────── Diagnosis History ───────────────────────────────

@app.get(
    "/patients/{patient_id}/history",
    response_model=DiagnosisHistoryResponse,
    tags=["Diagnosis"],
    summary="Get diagnosis history for a patient",
)
async def diagnosis_history(
    patient_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    # Verify patient exists
    await patient_manager.get(db, patient_id)

    query = (
        select(DiagnosisORM)
        .where(DiagnosisORM.patient_id == patient_id)
        .order_by(desc(DiagnosisORM.created_at))
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    items = [
        DiagnosisHistoryItem(
            diagnosis_id=r.diagnosis_id,
            timestamp=r.created_at,
            primary_diagnosis=r.primary_diagnosis,
            severity=r.severity,
            confidence=r.confidence,
            requesting_physician=r.requesting_physician,
        )
        for r in rows
    ]
    return DiagnosisHistoryResponse(
        patient_id=patient_id, total=len(items), items=items
    )


@app.get(
    "/diagnoses/{diagnosis_id}",
    response_model=DiagnosisReport,
    tags=["Diagnosis"],
    summary="Retrieve a specific diagnosis report",
)
async def get_diagnosis(diagnosis_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DiagnosisORM).where(DiagnosisORM.diagnosis_id == diagnosis_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail=f"Diagnosis {diagnosis_id} not found")

    # Fetch patient
    patient = await patient_manager.get(db, row.patient_id)

    return DiagnosisReport(
        diagnosis_id=row.diagnosis_id,
        patient_id=row.patient_id,
        patient_name=f"{patient.first_name} {patient.last_name}",
        timestamp=row.created_at,
        image_analysis=ImageAnalysisResult.model_validate_json(row.image_analysis_json)
        if row.image_analysis_json
        else None,
        symptom_analysis=SymptomAnalysisResult.model_validate_json(row.symptom_analysis_json)
        if row.symptom_analysis_json
        else None,
        primary_diagnosis=row.primary_diagnosis,
        confidence=row.confidence,
        severity=row.severity,
        differential_diagnoses=json.loads(row.differential_diagnoses or "[]"),
        clinical_summary=row.clinical_summary or "",
        findings=row.findings or "",
        assessment=row.assessment or "",
        plan=row.plan or "",
        follow_up=row.follow_up or "",
        requesting_physician=row.requesting_physician,
        processing_time_ms=row.processing_time_ms or 0.0,
    )


# ─────────────────────────── Stats ───────────────────────────────────────────

@app.get("/stats", tags=["System"], summary="Quick dashboard statistics")
async def get_stats(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func

    total_patients = (await db.execute(
        select(func.count()).select_from(PatientORM)
    )).scalar_one()

    total_diagnoses = (await db.execute(
        select(func.count()).select_from(DiagnosisORM)
    )).scalar_one()

    # Count by severity
    severity_rows = (await db.execute(
        select(DiagnosisORM.severity, func.count().label("cnt"))
        .group_by(DiagnosisORM.severity)
    )).all()
    severity_counts = {r.severity: r.cnt for r in severity_rows}

    return {
        "total_patients": total_patients,
        "total_diagnoses": total_diagnoses,
        "severity_breakdown": severity_counts,
        "system_version": settings.APP_VERSION,
        "demo_mode": settings.DEMO_MODE,
    }


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.server:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
    )
