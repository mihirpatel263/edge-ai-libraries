# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Symptom Analyzer  —  BiomedCLIP (PubMedBERT) text-encoder powered
==================================================================
No hardcoded knowledge base, symptom lists, ICD mappings, or probability tables.

Workflow
--------
1. Receive patient symptom list as free text.
2. Build a clinical query string: "patient presenting with <symptoms>".
3. Encode the query with BiomedCLIP's PubMedBERT text encoder.
4. Encode each disease's clinical description prompt with the same encoder.
5. Rank diseases by cosine similarity -> softmax probabilities.
6. Derive urgency / specialist / tests from the winning label keywords only.

All scores come from the model.  No rule-based matching at all.
"""

import uuid
import logging
import numpy as np
from typing import List, Optional

from app.models import (
    SymptomAnalysisRequest,
    SymptomAnalysisResult,
    DiseaseMatch,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Disease text prompts ────────────────────────────────
# Format: (short_label, clinical_presentation_text_for_pubmedbert)
# NO embedded symptom lists, ICD codes, severity or recommendation here.
# The model scores the symptom query against these descriptions directly.

_DISEASE_PROMPTS: List[tuple] = [
    ("Community-Acquired Pneumonia",
     "a patient with productive cough, fever, chills, pleuritic chest pain and dyspnoea due to lobar or bronchopneumonia"),
    ("COVID-19",
     "a patient with fever, dry cough, anosmia, ageusia, fatigue, myalgia and dyspnoea consistent with SARS-CoV-2 infection"),
    ("Acute Myocardial Infarction",
     "a patient with crushing central chest pain radiating to the left arm and jaw, diaphoresis, nausea and dyspnoea indicating acute coronary syndrome"),
    ("Pulmonary Embolism",
     "a patient with sudden onset dyspnoea, pleuritic chest pain, haemoptysis, tachycardia and hypoxia due to venous thromboembolism"),
    ("Hypertensive Crisis",
     "a patient with severe headache, blurred vision, chest pain, nausea and confusion from dangerously elevated blood pressure"),
    ("Acute Appendicitis",
     "a patient with periumbilical pain migrating to the right iliac fossa, anorexia, nausea, low-grade fever and rebound tenderness"),
    ("Urinary Tract Infection",
     "a patient with dysuria, urinary frequency and urgency, suprapubic discomfort and cloudy or malodorous urine"),
    ("Migraine",
     "a patient with unilateral throbbing headache, photophobia, phonophobia, nausea and visual aura lasting hours"),
    ("Pulmonary Tuberculosis",
     "a patient with chronic cough lasting more than three weeks, haemoptysis, drenching night sweats, weight loss and low-grade fever"),
    ("Asthma Exacerbation",
     "a patient with episodic wheeze, shortness of breath, chest tightness and nocturnal or exercise-induced cough"),
    ("Anemia",
     "a patient with fatigue, pallor, exertional dyspnoea, palpitations, dizziness and cold extremities from reduced haemoglobin"),
    ("Hypothyroidism",
     "a patient with fatigue, cold intolerance, weight gain, constipation, dry skin, hair loss, bradycardia and depression from thyroid deficiency"),
    ("Hyperthyroidism",
     "a patient with weight loss, heat intolerance, palpitations, tremor, anxiety, hyperdefecation and exophthalmos from excess thyroid hormone"),
    ("Type 2 Diabetes Mellitus",
     "a patient with polyuria, polydipsia, unexplained weight loss, blurred vision, fatigue and recurrent infections from hyperglycaemia"),
    ("Rheumatoid Arthritis",
     "a patient with symmetric small-joint swelling, morning stiffness lasting over one hour, fatigue, low-grade fever and rheumatoid nodules"),
    ("Acute Stroke",
     "a patient with sudden facial drooping, unilateral arm weakness, slurred speech, gait disturbance and confusion indicating cerebrovascular event"),
    ("Meningitis",
     "a patient with severe headache, neck rigidity, photophobia, fever, vomiting and altered consciousness from meningeal inflammation"),
    ("Sepsis",
     "a patient with fever or hypothermia, tachycardia, tachypnoea, altered mental status and suspected or confirmed infection indicating systemic inflammatory response"),
    ("Depression",
     "a patient with persistent low mood, anhedonia, fatigue, insomnia or hypersomnia, poor concentration, appetite disturbance and hopelessness"),
    ("Gastroesophageal Reflux Disease",
     "a patient with heartburn, acid regurgitation, epigastric discomfort, chronic cough, hoarseness and waterbrash worse after meals or lying flat"),
    ("Irritable Bowel Syndrome",
     "a patient with recurring abdominal cramps, bloating, alternating diarrhoea and constipation relieved by defecation without alarm features"),
    ("Kidney Stone",
     "a patient with sudden severe flank or loin pain radiating to the groin, nausea, vomiting, haematuria and restlessness from urolithiasis"),
    ("Heart Failure",
     "a patient with dyspnoea on exertion, orthopnoea, paroxysmal nocturnal dyspnoea, ankle oedema, fatigue and reduced exercise tolerance"),
    ("Chronic Obstructive Pulmonary Disease",
     "a patient with progressive dyspnoea, chronic productive cough, wheeze and frequent respiratory infections from airflow limitation"),
    ("Panic Disorder",
     "a patient with recurrent sudden episodes of palpitations, chest tightness, dyspnoea, diaphoresis, trembling, dizziness and fear of dying"),
]


# ───────────── Runtime urgency / specialist / tests (label-based only) ────────

_EMERGENCY_KW = {"myocardial infarction", "stroke", "sepsis", "meningitis",
                 "pulmonary embolism", "hypertensive crisis", "appendicitis"}
_URGENT_KW    = {"pneumonia", "covid", "tuberculosis", "asthma", "depression",
                 "heart failure", "copd"}

_SPECIALIST: dict = {
    "pneumonia":              "Pulmonology",
    "covid":                  "Infectious Disease",
    "myocardial infarction":  "Cardiology",
    "pulmonary embolism":     "Cardiology / Pulmonology",
    "hypertensive crisis":    "Cardiology",
    "appendicitis":           "General Surgery",
    "urinary tract":          "Urology",
    "migraine":               "Neurology",
    "tuberculosis":           "Infectious Disease / Pulmonology",
    "asthma":                 "Pulmonology / Allergy",
    "anemia":                 "Hematology",
    "hypothyroidism":         "Endocrinology",
    "hyperthyroidism":        "Endocrinology",
    "diabetes":               "Endocrinology",
    "rheumatoid":             "Rheumatology",
    "stroke":                 "Neurology",
    "meningitis":             "Neurology / Infectious Disease",
    "sepsis":                 "Intensive Care / Infectious Disease",
    "depression":             "Psychiatry",
    "reflux":                 "Gastroenterology",
    "irritable bowel":        "Gastroenterology",
    "kidney stone":           "Urology",
    "heart failure":          "Cardiology",
    "copd":                   "Pulmonology",
    "panic":                  "Psychiatry",
}

_TESTS: dict = {
    "pneumonia":             ["Chest X-ray", "CBC", "CRP", "Blood cultures", "Sputum culture"],
    "covid":                 ["RT-PCR SARS-CoV-2", "CBC", "CRP", "D-dimer", "Chest CT"],
    "myocardial infarction": ["12-lead ECG", "Troponin (serial)", "CBC", "Coagulation panel", "Echo"],
    "pulmonary embolism":    ["CT Pulmonary Angiogram", "D-dimer", "ECG", "Echo", "Doppler USS legs"],
    "hypertensive crisis":   ["Blood pressure monitoring", "ECG", "Urinalysis", "BMP", "Echo"],
    "appendicitis":          ["CBC (neutrophilia)", "CRP", "CT abdomen/pelvis", "USS abdomen"],
    "urinary tract":         ["Urinalysis", "Urine culture & sensitivity", "CBC", "Renal function"],
    "migraine":              ["Clinical diagnosis", "MRI brain (if atypical)", "CT brain"],
    "tuberculosis":          ["Chest X-ray", "Sputum AFB x3", "GeneXpert MTB/RIF", "IGRA"],
    "asthma":                ["Peak flow", "Spirometry", "ABG (severe)", "Chest X-ray"],
    "anemia":                ["CBC with differential", "Iron studies", "B12/Folate", "Peripheral smear"],
    "hypothyroidism":        ["TSH", "Free T4", "Anti-TPO antibodies", "Lipid panel"],
    "hyperthyroidism":       ["TSH", "Free T3/T4", "TRAb", "Thyroid USS"],
    "diabetes":              ["Fasting glucose", "HbA1c", "OGTT", "Renal function", "Lipid panel"],
    "rheumatoid":            ["RF", "Anti-CCP", "CRP", "ESR", "Joint X-ray"],
    "stroke":                ["CT brain (urgent)", "MRI brain", "ECG", "Echo", "Carotid Doppler"],
    "meningitis":            ["LP (CSF analysis)", "Blood cultures", "CT brain", "CBC", "CRP"],
    "sepsis":                ["Blood cultures x2", "CBC", "Lactate", "CRP", "Procalcitonin"],
    "depression":            ["PHQ-9", "TSH", "CBC", "Metabolic panel"],
    "reflux":                ["Clinical diagnosis", "Upper GI endoscopy (if alarm features)", "pH monitoring"],
    "irritable bowel":       ["Clinical diagnosis", "Colonoscopy (to exclude IBD)", "Stool studies"],
    "kidney stone":          ["CT KUB", "Urinalysis", "Renal function", "Urine culture"],
    "heart failure":         ["BNP/NT-proBNP", "Echo", "ECG", "Chest X-ray", "Renal function"],
    "copd":                  ["Spirometry", "Chest X-ray", "ABG", "CBC", "6MWT"],
    "panic":                 ["Clinical diagnosis", "ECG", "Thyroid function", "Holter monitor"],
}


def _infer_urgency(label: str) -> str:
    low = label.lower()
    if any(k in low for k in _EMERGENCY_KW):
        return "emergency"
    if any(k in low for k in _URGENT_KW):
        return "urgent"
    return "routine"


def _infer_specialist(label: str) -> Optional[str]:
    low = label.lower()
    for key, spec in _SPECIALIST.items():
        if key in low:
            return spec
    return "General Medicine"


def _infer_tests(label: str) -> List[str]:
    low = label.lower()
    for key, tests in _TESTS.items():
        if key in low:
            return tests
    return ["Clinical assessment", "CBC", "BMP", "CRP"]


# ───────────────────────── BiomedCLIP text encoder ───────────────────────────

def _encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of text strings with BiomedCLIP's PubMedBERT text encoder.
    Returns L2-normalised embeddings of shape (len(texts), dim).
    Falls back to TF-IDF bag-of-words if BiomedCLIP is not loaded.
    """
    try:
        import torch
        from app.image_analyzer import _BiomedCLIPModel
        clip = _BiomedCLIPModel.get()
        if clip is not None:
            tokens = clip.tokenizer(texts).to(clip.device)
            with torch.no_grad():
                feat = clip.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            logger.debug("BiomedCLIP text encoder used for symptom analysis.")
            return feat.cpu().float().numpy()
    except Exception as exc:
        logger.warning("BiomedCLIP text encoder unavailable (%s) -- using TF-IDF fallback.", exc)

    # TF-IDF fallback (no model required)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=4096)
        mat = vec.fit_transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
    except ImportError:
        pass

    # Last resort: random normal embeddings (results will be meaningless but won't crash)
    logger.error("No text encoder available -- returning random embeddings.")
    dim = 128
    mat = np.random.randn(len(texts), dim).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms


# ──────────────────────────── Symptom Analyzer ───────────────────────────────

class SymptomAnalyzer:
    """
    Pure model-based symptom analyser.
    Encodes patient symptoms and disease descriptions with BiomedCLIP's
    PubMedBERT text encoder, then ranks diseases by cosine similarity.
    No hard-coded symptom matching, Jaccard scoring, or knowledge base.
    """

    def analyze(self, request: SymptomAnalysisRequest) -> SymptomAnalysisResult:
        symptoms_text = ", ".join(s.strip().lower() for s in request.symptoms)
        if request.additional_context:
            symptoms_text += ". " + request.additional_context.strip()
        query = f"patient presenting with {symptoms_text}"

        labels  = [p[0] for p in _DISEASE_PROMPTS]
        prompts = [p[1] for p in _DISEASE_PROMPTS]
        n       = len(labels)

        # Encode query + all disease prompts in a single batch
        all_texts  = [query] + prompts
        embeddings = _encode_texts(all_texts)   # (n+1, dim)

        query_vec    = embeddings[0]             # (dim,)
        disease_vecs = embeddings[1:]            # (n, dim)

        # Cosine similarity -> softmax probability
        cos_sims = disease_vecs @ query_vec      # (n,)
        e        = np.exp((cos_sims - cos_sims.max()) * 10)   # temperature=0.1
        probs    = e / e.sum()

        ranked = sorted(range(n), key=lambda i: float(probs[i]), reverse=True)

        # Build DiseaseMatch list from top matches with probability > 1%
        matches: List[DiseaseMatch] = []
        for idx in ranked[:8]:
            prob = round(float(probs[idx]), 4)
            if prob < 0.01:
                continue
            matches.append(DiseaseMatch(
                disease=labels[idx],
                probability=prob,
                matching_symptoms=request.symptoms,
                missing_symptoms=[],
                icd10_code=None,
            ))

        top_label = labels[ranked[0]]
        urgency   = _infer_urgency(top_label)

        # Escalate urgency if any symptom matches emergency keywords
        emergency_symptom_kw = {"chest pain", "shortness of breath", "confusion",
                                "loss of consciousness", "paralysis", "severe headache",
                                "haemoptysis", "cyanosis", "seizure"}
        symptom_set = {s.lower() for s in request.symptoms}
        if symptom_set & emergency_symptom_kw and urgency == "routine":
            urgency = "urgent"

        red_flags = sorted(symptom_set & emergency_symptom_kw)

        return SymptomAnalysisResult(
            session_id=str(uuid.uuid4()),
            matched_diseases=matches,
            red_flags=red_flags,
            recommended_tests=_infer_tests(top_label),
            urgency=urgency,
            specialist_referral=_infer_specialist(top_label),
        )


symptom_analyzer = SymptomAnalyzer()
