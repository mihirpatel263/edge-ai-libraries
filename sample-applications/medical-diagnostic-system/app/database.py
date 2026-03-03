# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from app.config import settings

Base = declarative_base()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


# ─────────────────────────── ORM Models ──────────────────────────────────────

class PatientORM(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_uid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(String(20), nullable=False)
    gender = Column(String(20), nullable=False)
    blood_type = Column(String(10), nullable=True)
    allergies = Column(Text, nullable=True)
    chronic_conditions = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    emergency_contact = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    diagnoses = relationship("DiagnosisORM", back_populates="patient", cascade="all, delete-orphan")


class DiagnosisORM(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    primary_diagnosis = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    differential_diagnoses = Column(Text, nullable=True)  # JSON string
    clinical_summary = Column(Text, nullable=True)
    findings = Column(Text, nullable=True)
    assessment = Column(Text, nullable=True)
    plan = Column(Text, nullable=True)
    follow_up = Column(Text, nullable=True)
    requesting_physician = Column(String(200), nullable=True)
    image_analysis_json = Column(Text, nullable=True)   # stored as JSON
    symptom_analysis_json = Column(Text, nullable=True)  # stored as JSON
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    patient = relationship("PatientORM", back_populates="diagnoses")


# ─────────────────────────── DB helpers ──────────────────────────────────────

async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async session."""
    async with AsyncSessionLocal() as session:
        yield session
