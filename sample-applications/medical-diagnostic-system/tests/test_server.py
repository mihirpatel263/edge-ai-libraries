# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unit & integration tests for the Medical Diagnostic System.
Run with:  pytest tests/ -v
"""

import io
import json
import os
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

# conftest.py already sets env vars before this import
from app.database import engine, Base
from app.server import app
from app.symptom_analyzer import symptom_analyzer
from app.models import SymptomAnalysisRequest, ImageModality


# ─────────────────────────── Fixtures ────────────────────────────────────────

@pytest.fixture(autouse=True)
async def clean_db():
    """Drop and recreate all tables before every test for full isolation."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Cleanup after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client(clean_db):
    """ASGI test client — tables are always fresh via clean_db fixture."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a minimal valid JPEG in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
async def registered_patient(client):
    """Returns a freshly created patient dict."""
    resp = await client.post("/v1/meddiag/patients", json={
        "first_name": "Test",
        "last_name": "Patient",
        "date_of_birth": "1990-01-15",
        "gender": "female",
        "blood_type": "A+",
    })
    assert resp.status_code == 201
    return resp.json()


# ─────────────────────────── Health ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/v1/meddiag/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["demo_mode"] is True


# ─────────────────────────── Patients ────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_patient(client):
    resp = await client.post("/v1/meddiag/patients", json={
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": "1985-04-12",
        "gender": "female",
        "blood_type": "O+",
        "chronic_conditions": "Hypertension",
        "current_medications": "Lisinopril 10mg",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["first_name"] == "Jane"
    assert data["id"] >= 1
    assert "patient_uid" in data


@pytest.mark.asyncio
async def test_list_patients(client, registered_patient):
    resp = await client.get("/v1/meddiag/patients")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_get_patient(client, registered_patient):
    pid = registered_patient["id"]
    resp = await client.get(f"/v1/meddiag/patients/{pid}")
    assert resp.status_code == 200
    assert resp.json()["id"] == pid


@pytest.mark.asyncio
async def test_get_patient_not_found(client):
    resp = await client.get("/v1/meddiag/patients/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_patient(client, registered_patient):
    pid = registered_patient["id"]
    resp = await client.patch(f"/v1/meddiag/patients/{pid}", json={
        "allergies": "Penicillin",
        "notes": "Updated note"
    })
    assert resp.status_code == 200
    assert resp.json()["allergies"] == "Penicillin"


@pytest.mark.asyncio
async def test_delete_patient(client, registered_patient):
    pid = registered_patient["id"]
    resp = await client.delete(f"/v1/meddiag/patients/{pid}")
    assert resp.status_code == 200
    # Verify gone
    resp2 = await client.get(f"/v1/meddiag/patients/{pid}")
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_search_patients(client, registered_patient):
    resp = await client.get("/v1/meddiag/patients?search=Test")
    assert resp.status_code == 200


# ─────────────────────────── Symptom Analysis ────────────────────────────────

def test_symptom_analyzer_pneumonia():
    req = SymptomAnalysisRequest(
        patient_id=1,
        symptoms=["fever", "cough", "shortness of breath", "chest pain"],
        duration_days=5,
    )
    result = symptom_analyzer.analyze(req)
    assert len(result.matched_diseases) > 0
    assert result.urgency in ("routine", "urgent", "emergency")
    # Pneumonia or COVID-19 should be in top matches
    disease_names = [m.disease.lower() for m in result.matched_diseases]
    assert any("pneumonia" in d or "covid" in d for d in disease_names)


def test_symptom_analyzer_cardiac():
    req = SymptomAnalysisRequest(
        patient_id=1,
        symptoms=["chest pain", "chest pressure", "jaw pain", "sweating", "arm pain"],
    )
    result = symptom_analyzer.analyze(req)
    assert result.urgency == "emergency"
    assert len(result.red_flags) > 0


def test_symptom_analyzer_no_match():
    req = SymptomAnalysisRequest(
        patient_id=1,
        symptoms=["xyz_unknown_symptom_12345"],
    )
    result = symptom_analyzer.analyze(req)
    assert len(result.matched_diseases) == 0


@pytest.mark.asyncio
async def test_api_symptom_analysis(client, registered_patient):
    pid = registered_patient["id"]
    resp = await client.post("/v1/meddiag/analyze/symptoms", json={
        "patient_id": pid,
        "symptoms": ["fever", "cough", "fatigue"],
        "duration_days": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "matched_diseases" in data
    assert "urgency" in data
    assert "session_id" in data


@pytest.mark.asyncio
async def test_symptom_analysis_invalid_patient(client):
    resp = await client.post("/v1/meddiag/analyze/symptoms", json={
        "patient_id": 99999,
        "symptoms": ["fever"],
    })
    assert resp.status_code == 404


# ─────────────────────────── Image Analysis ──────────────────────────────────

@pytest.mark.asyncio
async def test_image_analysis(client, registered_patient, sample_image_bytes):
    pid = registered_patient["id"]
    resp = await client.post(
        "/v1/meddiag/analyze/image",
        data={
            "patient_id": str(pid),
            "modality": "chest_xray",
            "clinical_notes": "Shortness of breath",
        },
        files={"image": ("xray.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "primary_diagnosis" in data
    assert "confidence_score" in data
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert "severity" in data
    assert "processing_time_ms" in data


@pytest.mark.asyncio
async def test_image_analysis_all_modalities(client, registered_patient, sample_image_bytes):
    pid = registered_patient["id"]
    for modality in ["chest_xray", "ct_scan", "mri", "dermatology", "pathology"]:
        resp = await client.post(
            "/v1/meddiag/analyze/image",
            data={"patient_id": str(pid), "modality": modality},
            files={"image": ("img.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200, f"Failed for modality {modality}: {resp.text}"


@pytest.mark.asyncio
async def test_image_analysis_missing_patient(client, sample_image_bytes):
    resp = await client.post(
        "/v1/meddiag/analyze/image",
        data={"patient_id": "99999", "modality": "chest_xray"},
        files={"image": ("img.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 404


# ─────────────────────────── Multimodal Diagnosis ────────────────────────────

@pytest.mark.asyncio
async def test_full_diagnosis_symptoms_only(client, registered_patient):
    pid = registered_patient["id"]
    resp = await client.post(
        "/v1/meddiag/diagnose",
        data={
            "patient_id": str(pid),
            "symptoms": "fever, cough, fatigue",
            "duration_days": "7",
            "requesting_physician": "Dr. Smith",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "diagnosis_id" in data
    assert "primary_diagnosis" in data
    assert "confidence" in data
    assert "clinical_summary" in data
    assert "findings" in data
    assert "assessment" in data
    assert "plan" in data
    assert "follow_up" in data


@pytest.mark.asyncio
async def test_full_diagnosis_with_image(client, registered_patient, sample_image_bytes):
    pid = registered_patient["id"]
    resp = await client.post(
        "/v1/meddiag/diagnose",
        data={
            "patient_id": str(pid),
            "symptoms": "chest pain, shortness of breath",
            "modality": "chest_xray",
            "requesting_physician": "Dr. Johnson",
        },
        files={"image": ("xray.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["image_analysis"] is not None
    assert data["symptom_analysis"] is not None


@pytest.mark.asyncio
async def test_full_diagnosis_no_input(client, registered_patient):
    pid = registered_patient["id"]
    resp = await client.post(
        "/v1/meddiag/diagnose",
        data={"patient_id": str(pid)},  # no symptoms, no image
    )
    assert resp.status_code == 422


# ─────────────────────────── History & Reports ───────────────────────────────

@pytest.mark.asyncio
async def test_diagnosis_history(client, registered_patient):
    pid = registered_patient["id"]
    # Create a diagnosis first
    await client.post(
        "/v1/meddiag/diagnose",
        data={"patient_id": str(pid), "symptoms": "fatigue, headache"},
    )
    resp = await client.get(f"/v1/meddiag/patients/{pid}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_id"] == pid
    assert data["total"] >= 1
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_get_diagnosis_by_id(client, registered_patient):
    pid = registered_patient["id"]
    create_resp = await client.post(
        "/v1/meddiag/diagnose",
        data={"patient_id": str(pid), "symptoms": "fever, nausea"},
    )
    assert create_resp.status_code == 201
    diagnosis_id = create_resp.json()["diagnosis_id"]

    resp = await client.get(f"/v1/meddiag/diagnoses/{diagnosis_id}")
    assert resp.status_code == 200
    assert resp.json()["diagnosis_id"] == diagnosis_id


@pytest.mark.asyncio
async def test_get_diagnosis_not_found(client):
    resp = await client.get("/v1/meddiag/diagnoses/non-existent-uuid")
    assert resp.status_code == 404


# ─────────────────────────── Stats ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats(client):
    resp = await client.get("/v1/meddiag/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_patients" in data
    assert "total_diagnoses" in data
    assert "severity_breakdown" in data
