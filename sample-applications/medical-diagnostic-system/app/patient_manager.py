# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from fastapi import HTTPException

from app.database import PatientORM
from app.models import PatientCreate, PatientUpdate, PatientResponse


class PatientManager:
    """CRUD operations for patient records."""

    async def create(self, db: AsyncSession, data: PatientCreate) -> PatientResponse:
        patient = PatientORM(
            patient_uid=str(uuid.uuid4()),
            first_name=data.first_name,
            last_name=data.last_name,
            date_of_birth=data.date_of_birth,
            gender=data.gender.value,
            blood_type=data.blood_type,
            allergies=data.allergies,
            chronic_conditions=data.chronic_conditions,
            current_medications=data.current_medications,
            emergency_contact=data.emergency_contact,
            notes=data.notes,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(patient)
        await db.commit()
        await db.refresh(patient)
        return PatientResponse.model_validate(patient)

    async def get(self, db: AsyncSession, patient_id: int) -> PatientResponse:
        result = await db.execute(select(PatientORM).where(PatientORM.id == patient_id))
        patient = result.scalar_one_or_none()
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        return PatientResponse.model_validate(patient)

    async def list_all(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
    ) -> List[PatientResponse]:
        query = select(PatientORM).order_by(PatientORM.created_at.desc())
        if search:
            term = f"%{search}%"
            query = query.where(
                (PatientORM.first_name.ilike(term))
                | (PatientORM.last_name.ilike(term))
                | (PatientORM.patient_uid.ilike(term))
            )
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        patients = result.scalars().all()
        return [PatientResponse.model_validate(p) for p in patients]

    async def count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count()).select_from(PatientORM))
        return result.scalar_one()

    async def update(
        self, db: AsyncSession, patient_id: int, data: PatientUpdate
    ) -> PatientResponse:
        result = await db.execute(select(PatientORM).where(PatientORM.id == patient_id))
        patient = result.scalar_one_or_none()
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

        for field, value in data.model_dump(exclude_none=True).items():
            setattr(patient, field, value)
        patient.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(patient)
        return PatientResponse.model_validate(patient)

    async def delete(self, db: AsyncSession, patient_id: int) -> dict:
        result = await db.execute(select(PatientORM).where(PatientORM.id == patient_id))
        patient = result.scalar_one_or_none()
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        await db.delete(patient)
        await db.commit()
        return {"message": f"Patient {patient_id} deleted successfully"}


patient_manager = PatientManager()
