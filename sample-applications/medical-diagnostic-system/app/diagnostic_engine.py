# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Diagnostic Engine
=============================
Fuses image-analysis results with symptom-analysis results to produce a
final diagnosis.  The fusion strategy is a simple confidence-weighted
Bayesian combination: when both modalities point to the same condition
the evidence is reinforced; when they conflict the higher-confidence
source takes precedence.
"""

import uuid
import time
from datetime import datetime, timezone
from typing import Optional, List

from app.models import (
    MultimodalDiagnosisRequest,
    DiagnosisReport,
    ImageAnalysisResult,
    SymptomAnalysisResult,
    PatientResponse,
    Severity,
)
from app.report_generator import report_generator


_SEVERITY_ORDER = {
    Severity.NONE: 0,
    Severity.MILD: 1,
    Severity.MODERATE: 2,
    Severity.SEVERE: 3,
    Severity.CRITICAL: 4,
}


class DiagnosticEngine:
    """Combines multimodal signals and produces a DiagnosisReport."""

    async def diagnose(
        self,
        patient: PatientResponse,
        image_result: Optional[ImageAnalysisResult],
        symptom_result: Optional[SymptomAnalysisResult],
        requesting_physician: Optional[str] = None,
    ) -> DiagnosisReport:
        start = time.perf_counter()

        primary_diagnosis, confidence, severity = self._fuse(image_result, symptom_result)
        differentials = self._collect_differentials(image_result, symptom_result, primary_diagnosis)

        report_sections = await report_generator.generate(
            patient=patient,
            image_result=image_result,
            symptom_result=symptom_result,
            primary_diagnosis=primary_diagnosis,
            severity=severity,
            confidence=confidence,
            requesting_physician=requesting_physician,
        )

        elapsed = round((time.perf_counter() - start) * 1000, 2)

        return DiagnosisReport(
            diagnosis_id=str(uuid.uuid4()),
            patient_id=patient.id,
            patient_name=f"{patient.first_name} {patient.last_name}",
            timestamp=datetime.now(timezone.utc),
            image_analysis=image_result,
            symptom_analysis=symptom_result,
            primary_diagnosis=primary_diagnosis,
            confidence=round(confidence, 4),
            severity=severity,
            differential_diagnoses=differentials,
            clinical_summary=report_sections["clinical_summary"],
            findings=report_sections["findings"],
            assessment=report_sections["assessment"],
            plan=report_sections["plan"],
            follow_up=report_sections["follow_up"],
            requesting_physician=requesting_physician,
            processing_time_ms=elapsed,
        )

    # ─────────────────────────── Fusion logic ────────────────────────────────

    def _fuse(
        self,
        img: Optional[ImageAnalysisResult],
        sym: Optional[SymptomAnalysisResult],
    ) -> tuple[str, float, Severity]:
        """Return (primary_diagnosis, confidence, severity)."""

        # ── Only image ────────────────────────────────────────────────────────
        if img and not sym:
            return img.primary_diagnosis, img.confidence_score, img.severity

        # ── Only symptoms ─────────────────────────────────────────────────────
        if sym and not img:
            top = sym.matched_diseases[0] if sym.matched_diseases else None
            if top:
                severity = self._urgency_to_severity(sym.urgency)
                return top.disease, top.probability, severity
            return "Undetermined", 0.0, Severity.NONE

        # ── Both modalities ───────────────────────────────────────────────────
        if img and sym and sym.matched_diseases:
            img_diag = img.primary_diagnosis.lower()
            sym_top = sym.matched_diseases[0]

            # Check if they agree (simple keyword overlap)
            agree = any(
                word in sym_top.disease.lower()
                for word in img_diag.split()
                if len(word) > 4
            )

            if agree:
                # Reinforce both signals
                combined_conf = min(
                    0.99,
                    img.confidence_score * 0.6 + sym_top.probability * 0.4 + 0.08,
                )
                severity = max(
                    [img.severity, self._urgency_to_severity(sym.urgency)],
                    key=lambda s: _SEVERITY_ORDER[s],
                )
                return img.primary_diagnosis, round(combined_conf, 4), severity
            else:
                # Pick the higher-confidence source
                if img.confidence_score >= sym_top.probability:
                    return img.primary_diagnosis, img.confidence_score, img.severity
                else:
                    sev = self._urgency_to_severity(sym.urgency)
                    return sym_top.disease, sym_top.probability, sev

        return "Evaluation in Progress", 0.5, Severity.NONE

    @staticmethod
    def _urgency_to_severity(urgency: str) -> Severity:
        return {
            "emergency": Severity.CRITICAL,
            "urgent": Severity.MODERATE,
            "routine": Severity.MILD,
        }.get(urgency, Severity.NONE)

    @staticmethod
    def _collect_differentials(
        img: Optional[ImageAnalysisResult],
        sym: Optional[SymptomAnalysisResult],
        primary: str,
    ) -> List[str]:
        seen = set()
        result = []

        if img:
            for d in img.differential_diagnoses:
                if d.lower() != primary.lower() and d not in seen:
                    seen.add(d)
                    result.append(d)

        if sym:
            for m in sym.matched_diseases[1:5]:
                if m.disease.lower() != primary.lower() and m.disease not in seen:
                    seen.add(m.disease)
                    result.append(m.disease)

        return result[:6]


diagnostic_engine = DiagnosticEngine()
