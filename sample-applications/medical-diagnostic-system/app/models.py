# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


# ─────────────────────────── Enumerations ────────────────────────────────────

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ImageModality(str, Enum):
    CHEST_XRAY = "chest_xray"
    CT_SCAN = "ct_scan"
    MRI = "mri"
    DERMATOLOGY = "dermatology"
    PATHOLOGY = "pathology"
    OTHER = "other"


class DiagnosticStatus(str, Enum):
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    INCONCLUSIVE = "inconclusive"


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


# ─────────────────────────── Patient Schemas ─────────────────────────────────

class PatientBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: str
    gender: Gender
    blood_type: Optional[str] = None
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    emergency_contact: Optional[str] = None
    notes: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "first_name": "Jane", "last_name": "Doe",
                "date_of_birth": "1985-04-12", "gender": "female",
                "blood_type": "A+", "allergies": "Penicillin",
                "chronic_conditions": "Hypertension",
                "current_medications": "Lisinopril 10mg",
            }
        }
    }


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    notes: Optional[str] = None


class PatientResponse(PatientBase):
    id: int
    patient_uid: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────── Image Analysis Schemas ──────────────────────────

class ImageAnalysisRequest(BaseModel):
    patient_id: int
    modality: ImageModality
    body_part: Optional[str] = None
    clinical_notes: Optional[str] = None


class Finding(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    region: Optional[str] = None


class ClassProbability(BaseModel):
    label: str
    probability: float
    icd10: Optional[str] = None


class VLMReportSection(BaseModel):
    """Structured report produced by the LLaVA vision-language model."""
    findings: str = ""
    primary_diagnosis: str = ""
    severity: str = ""
    differential_diagnoses: str = ""
    treatment: str = ""
    raw: str = ""
    model_id: str = ""


class ImageAnalysisResult(BaseModel):
    image_id: str
    modality: ImageModality
    body_region: Optional[str] = None
    status: DiagnosticStatus
    findings: List[Finding]
    primary_diagnosis: str
    differential_diagnoses: List[str]
    severity: Severity
    recommendation: str
    confidence_score: float
    processing_time_ms: float
    # Full ranked probability distribution from BiomedCLIP inference
    inference_source: str = "demo"   # "biomedclip" | "openvino" | "demo"
    class_probabilities: List[ClassProbability] = []
    # Natural-language report from LLaVA VLM (None when model unavailable)
    vlm_report: Optional[VLMReportSection] = None


# ─────────────────────────── Symptom Analysis Schemas ────────────────────────

class SymptomAnalysisRequest(BaseModel):
    patient_id: int
    symptoms: List[str] = Field(..., min_length=1)
    duration_days: Optional[int] = Field(None, ge=0)
    onset: Optional[str] = None
    severity_self_reported: Optional[str] = None
    additional_history: Optional[str] = None


class DiseaseMatch(BaseModel):
    disease: str
    probability: float = Field(..., ge=0.0, le=1.0)
    matching_symptoms: List[str]
    missing_symptoms: List[str]
    icd10_code: Optional[str] = None


class SymptomAnalysisResult(BaseModel):
    session_id: str
    matched_diseases: List[DiseaseMatch]
    red_flags: List[str]
    recommended_tests: List[str]
    urgency: str  # "routine" | "urgent" | "emergency"
    specialist_referral: Optional[str] = None


# ─────────────────────────── Multimodal Diagnosis Schemas ────────────────────

class MultimodalDiagnosisRequest(BaseModel):
    patient_id: int
    image_analysis_id: Optional[str] = None
    symptoms: Optional[List[str]] = None
    duration_days: Optional[int] = None
    additional_history: Optional[str] = None
    requesting_physician: Optional[str] = None


class DiagnosisReport(BaseModel):
    diagnosis_id: str
    patient_id: int
    patient_name: str
    timestamp: datetime
    # Clinical inputs
    image_analysis: Optional[ImageAnalysisResult] = None
    symptom_analysis: Optional[SymptomAnalysisResult] = None
    # Final assessment
    primary_diagnosis: str
    confidence: float
    severity: Severity
    differential_diagnoses: List[str]
    # Report content
    clinical_summary: str
    findings: str
    assessment: str
    plan: str
    follow_up: str
    requesting_physician: Optional[str] = None
    # Metadata
    model_version: str = "1.0.0"
    processing_time_ms: float


# ─────────────────────────── History Schemas ─────────────────────────────────

class DiagnosisHistoryItem(BaseModel):
    diagnosis_id: str
    timestamp: datetime
    primary_diagnosis: str
    severity: Severity
    confidence: float
    requesting_physician: Optional[str] = None

    model_config = {"from_attributes": True}


class DiagnosisHistoryResponse(BaseModel):
    patient_id: int
    total: int
    items: List[DiagnosisHistoryItem]


# ─────────────────────────── Health check ────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    demo_mode: bool
    llm_available: bool
    openvino_available: bool
    biomedclip_available: bool = False
    database: str
