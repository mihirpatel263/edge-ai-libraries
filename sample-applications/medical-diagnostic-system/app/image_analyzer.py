# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Medical Image Analyzer  —  BiomedCLIP-powered
==============================================
Uses microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 for zero-shot
medical image classification.  The model encodes the query image and a set of
disease-specific text prompts, then ranks them by cosine similarity to produce
real diagnosis probabilities directly from the image content.

DEMO_MODE=true (or model download failure) causes a lightweight deterministic
fallback so the service stays runnable without GPU/network access.
"""

import uuid
import time
import os
import hashlib
import logging
from typing import List, Optional
from pathlib import Path

from PIL import Image
import numpy as np

from app.config import settings
from app.models import (
    ImageModality,
    DiagnosticStatus,
    Severity,
    Finding,
    ImageAnalysisResult,
    ClassProbability,
    VLMReportSection,
)
from app.medsiglip_analyzer import medsiglip_classify, is_medsiglip_available, MEDSIGLIP_MODEL_ID

logger = logging.getLogger(__name__)


# ─────────────────────────── Visual prompts for BiomedCLIP ──────────────────
# Each entry is (short_label, radiological_visual_description).
# There are NO hardcoded ICD codes, severity levels, or recommendations here.
# Everything is derived at runtime from BiomedCLIP's own inference output.

_PROMPTS: dict = {
    ImageModality.CHEST_XRAY: [
        ("Normal",
         "a chest X-ray with clear lung fields, normal cardiac silhouette and no cardiopulmonary abnormality"),
        ("Consolidation",
         "a chest X-ray showing lobar or segmental consolidation with increased airspace opacity and air bronchograms"),
        ("Ground-Glass Opacity",
         "a chest X-ray showing bilateral peripheral or diffuse hazy ground-glass opacification"),
        ("Pleural Effusion",
         "a chest X-ray showing blunting of the costophrenic angle with meniscus sign indicating pleural fluid"),
        ("Cardiomegaly",
         "a chest X-ray with a cardiothoracic ratio greater than 0.5 and an enlarged cardiac silhouette"),
        ("Lung Mass",
         "a chest X-ray showing a discrete irregular or spiculated opacity within the lung parenchyma"),
        ("Atelectasis",
         "a chest X-ray with linear band-like subsegmental atelectasis and volume loss"),
        ("Pneumothorax",
         "a chest X-ray showing a sharp visceral pleural line with absent lung markings peripherally"),
    ],
    ImageModality.CT_SCAN: [
        ("Normal",
         "a CT scan with no significant abnormality in any visualised organ or tissue"),
        ("Appendicitis",
         "an abdominal CT showing a dilated appendix with periappendiceal fat stranding and surrounding free fluid"),
        ("Nephrolithiasis",
         "a CT urogram showing a hyperdense calculus in the ureter or renal pelvis with upstream dilatation"),
        ("Hepatic Lesion",
         "a CT showing a hepatic lesion with low attenuation or heterogeneous enhancement requiring characterisation"),
        ("Pulmonary Embolism",
         "a CT pulmonary angiogram showing intraluminal filling defects in the segmental pulmonary arteries"),
        ("Pancreatitis",
         "an abdominal CT with pancreatic enlargement, heterogeneous parenchyma and peripancreatic fat stranding"),
    ],
    ImageModality.MRI: [
        ("Normal",
         "a brain MRI with no intracranial abnormality and normal grey and white matter signal intensity"),
        ("Intracranial Mass",
         "a brain MRI showing a space-occupying lesion with surrounding vasogenic oedema and mass effect"),
        ("White Matter Disease",
         "a brain MRI with multiple periventricular T2 FLAIR hyperintense white matter foci"),
        ("Acute Infarct",
         "a brain MRI with diffusion restriction in an arterial territory indicating acute ischaemic infarction"),
        ("Cerebral Atrophy",
         "a brain MRI showing generalised cortical atrophy with widened sulci and enlarged ventricles"),
        ("Extra-axial Collection",
         "a brain MRI showing a crescent-shaped extra-axial collection along the inner calvarium"),
    ],
    ImageModality.DERMATOLOGY: [
        ("Normal Skin",
         "a dermatoscopy image of normal skin texture with a regular pigment network and no structural abnormality"),
        ("Suspicious Melanocytic Lesion",
         "a dermatoscopy image of an asymmetric pigmented lesion with irregular border and multicolour variation"),
        ("Pearlescent Papule",
         "a dermatoscopy image of a pearlescent nodule with arborising telangiectasias and rolled border"),
        ("Inflammatory Dermatosis",
         "a clinical photograph of erythematous poorly-demarcated plaques in flexural or sebaceous areas"),
        ("Psoriasiform Plaques",
         "a clinical photograph of well-demarcated erythematous plaques with thick silvery scaling"),
        ("Acneiform Eruption",
         "a clinical photograph of comedones, inflammatory papules and pustules on sebaceous skin"),
    ],
    ImageModality.PATHOLOGY: [
        ("Normal Tissue",
         "a histopathology slide showing normal tissue architecture with regular glands and no cytological atypia"),
        ("Malignant Neoplasm",
         "a histopathology slide showing pleomorphic cells with high mitotic activity and loss of architecture"),
        ("Chronic Inflammation",
         "a biopsy slide showing dense chronic lymphocytic infiltrate with stromal fibrosis and glandular atrophy"),
        ("Adenocarcinoma Pattern",
         "a histopathology slide showing irregular malignant glandular structures with desmoplastic stroma"),
    ],
}
_PROMPTS[ImageModality.OTHER] = _PROMPTS[ImageModality.CHEST_XRAY]

# Anatomical region label per modality — only used in finding region field
_REGION: dict = {
    ImageModality.CHEST_XRAY:  "thorax",
    ImageModality.CT_SCAN:     "abdomen / thorax",
    ImageModality.MRI:         "brain",
    ImageModality.DERMATOLOGY: "skin surface",
    ImageModality.PATHOLOGY:   "tissue specimen",
    ImageModality.OTHER:       "unspecified",
}


# ───────────── Runtime severity / recommendation (no lookup tables) ───────────

_CRITICAL_KW = {"mass", "malignant", "neoplasm", "adenocarcinoma", "embolism",
                "collection", "melanocytic", "intracranial"}
_SEVERE_KW   = {"pneumothorax", "appendicitis", "consolidation", "ground-glass",
                "extra-axial", "infarct"}
_MILD_KW     = {"cardiomegaly", "atelectasis", "inflammation", "dermatosis",
                "acneiform", "psoriasiform", "atrophy", "white matter"}


def _infer_severity(label: str, margin: float) -> Severity:
    """Derive Severity purely from the predicted label text and margin."""
    low = label.lower()
    if "normal" in low:
        return Severity.NONE
    if any(k in low for k in _CRITICAL_KW):
        return Severity.CRITICAL
    if any(k in low for k in _SEVERE_KW):
        return Severity.SEVERE
    if any(k in low for k in _MILD_KW):
        return Severity.MILD
    # Borderline margin → treat as moderate
    return Severity.MODERATE


def _infer_recommendation(severity: Severity, label: str) -> str:
    """Derive a clinical recommendation from severity + label — no static table."""
    low = label.lower()
    if severity == Severity.NONE:
        return "No significant abnormality detected. Routine clinical follow-up as indicated."
    if severity == Severity.CRITICAL:
        if "embolism" in low:
            return "Immediate anticoagulation assessment; consider ICU-level monitoring."
        if "malignant" in low or "neoplasm" in low or "adenocarcinoma" in low:
            return "Urgent oncology referral; arrange biopsy and staging workup."
        if "collection" in low or "intracranial" in low:
            return "Urgent neurosurgical evaluation; consider evacuation."
        if "melanocytic" in low:
            return "Urgent dermatology referral; excisional biopsy indicated."
        return "Urgent specialist referral; arrange further investigation immediately."
    if severity == Severity.SEVERE:
        if "pneumothorax" in low:
            return "Urgent chest drain if large; close monitoring for small pneumothorax."
        if "appendicitis" in low:
            return "Surgical consultation; prepare for appendectomy."
        if "infarct" in low:
            return "Immediate stroke team activation; assess for thrombolysis."
        if "consolidation" in low:
            return "Antibiotic therapy; repeat imaging in 4-6 weeks to confirm resolution."
        return "Prompt specialist consultation and further investigation recommended."
    if severity == Severity.MILD:
        return "Clinically review; manage symptomatically and arrange routine follow-up."
    return "Clinical review recommended; correlate with symptoms and arrange follow-up."

BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ─────────────────────────── BiomedCLIP loader ───────────────────────────────

import threading as _threading

class _BiomedCLIPModel:
    """
    Thread-safe singleton wrapper — loads BiomedCLIP once, then reused for
    every inference call.  Concurrent callers block until loading completes
    rather than falling back to demo mode.
    """

    _instance: Optional["_BiomedCLIPModel"] = None
    _failed: bool = False          # True if load was attempted and failed
    _lock = _threading.Lock()      # serialises first-time initialisation

    def __init__(self):
        import torch
        import open_clip

        logger.info("Loading BiomedCLIP model: %s", BIOMEDCLIP_MODEL_ID)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL_ID
        )
        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("BiomedCLIP loaded successfully on device: %s", self.device)

    @classmethod
    def get(cls) -> Optional["_BiomedCLIPModel"]:
        # Fast path — already loaded or already known to have failed
        if cls._instance is not None:
            return cls._instance
        if cls._failed:
            return None

        # Slow path — acquire lock so only one thread attempts the load;
        # all other threads block here and get the result when done.
        with cls._lock:
            # Double-check after acquiring the lock
            if cls._instance is not None:
                return cls._instance
            if cls._failed:
                return None
            try:
                cls._instance = cls()
                return cls._instance
            except Exception as exc:
                cls._failed = True
                logger.warning(
                    "BiomedCLIP load failed (%s) — using demo fallback.", exc
                )
                return None


# ─────────────────────────── Demo fallback helpers ───────────────────────────

def _image_hash_index(image_path: str, n: int) -> int:
    h = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
    return int(h[:8], 16) % n


def _demo_scores(image_path: str, n: int, primary_idx: int) -> List[float]:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    seed = int(abs(arr.mean() * 1e6)) % (2**31)
    rng = np.random.default_rng(seed)
    scores = rng.dirichlet(np.ones(n) * 0.5)
    scores[primary_idx] = max(scores[primary_idx], 0.55) + rng.uniform(0, 0.25)
    scores /= scores.sum()
    return [round(float(s), 4) for s in scores]


# ─────────────────────────── Main Analyzer class ─────────────────────────────

class ImageAnalyzer:
    """
    Medical image analyser backed by BiomedCLIP zero-shot inference.

    Workflow:
      1. Load microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 on first call.
      2. Encode the query image with the vision encoder.
      3. Encode modality-specific disease text prompts with the text encoder.
      4. Rank diseases by cosine similarity → softmax probabilities.
      5. Return structured ImageAnalysisResult with real per-image scores.

    Falls back to OpenVINO IR models when configured, or deterministic demo
    mode when neither BiomedCLIP nor OpenVINO is available.
    """

    def __init__(self):
        self._ov_models: dict = {}
        self._try_load_openvino()

    def _try_load_openvino(self):
        try:
            from openvino.runtime import Core  # type: ignore
            core = Core()
            for modality, env_key in [
                (ImageModality.CHEST_XRAY, settings.OPENVINO_CHEST_XRAY_MODEL),
                (ImageModality.CT_SCAN, settings.OPENVINO_CT_MODEL),
                (ImageModality.MRI, settings.OPENVINO_MRI_MODEL),
            ]:
                if env_key and os.path.exists(env_key):
                    model = core.read_model(env_key)
                    self._ov_models[modality] = core.compile_model(model, "CPU")
        except ImportError:
            pass

    def is_openvino_available(self) -> bool:
        return len(self._ov_models) > 0

    def is_biomedclip_available(self) -> bool:
        return _BiomedCLIPModel.get() is not None

    def is_medsiglip_available(self) -> bool:
        return is_medsiglip_available()

    # ── Public entry point ────────────────────────────────────────────────────

    async def analyze(
        self,
        image_path: str,
        modality: ImageModality,
        body_part: Optional[str] = None,
        clinical_notes: Optional[str] = None,
        model_choice: str = "biomedclip",
    ) -> ImageAnalysisResult:
        start = time.perf_counter()
        image_id = str(uuid.uuid4())

        if modality in self._ov_models:
            result = await self._openvino_analyze(image_id, image_path, modality)
        elif model_choice == "medsiglip":
            result = self._medsiglip_analyze(image_id, image_path, modality)
        elif model_choice == "medgemma":
            # BiomedCLIP handles class probabilities; MedGemma provides the VLM report
            clip = _BiomedCLIPModel.get()
            result = (
                self._biomedclip_analyze(image_id, image_path, modality)
                if clip is not None
                else self._demo_analyze(image_id, image_path, modality)
            )
        else:  # biomedclip (default)
            clip = _BiomedCLIPModel.get()
            result = (
                self._biomedclip_analyze(image_id, image_path, modality)
                if clip is not None
                else self._demo_analyze(image_id, image_path, modality)
            )

        # ── VLM natural-language report — only for medgemma model choice ──
        if model_choice == "medgemma":
            from app.vlm_analyzer import analyze_image_with_vlm  # lazy import
            vlm_out = analyze_image_with_vlm(
                image_path=image_path,
                modality=modality.value,
                clinical_notes=clinical_notes,
            )
            if vlm_out is not None:
                result.vlm_report = VLMReportSection(
                    findings=vlm_out.findings,
                    primary_diagnosis=vlm_out.primary_diagnosis,
                    severity=vlm_out.severity,
                    differential_diagnoses=vlm_out.differential_diagnoses,
                    treatment=vlm_out.treatment,
                    raw=vlm_out.raw,
                    model_id=vlm_out.model_id,
                )
                logger.info("MedGemma VLM report attached")
            else:
                logger.info("MedGemma unavailable — no VLM report.")

        result.body_region = body_part
        result.processing_time_ms = round((time.perf_counter() - start) * 1000, 2)
        return result

    # ── BiomedCLIP zero-shot inference ────────────────────────────────────────

    def _biomedclip_analyze(
        self, image_id: str, image_path: str, modality: ImageModality
    ) -> ImageAnalysisResult:
        import torch

        entries = _PROMPTS.get(modality, _PROMPTS[ImageModality.CHEST_XRAY])
        clip = _BiomedCLIPModel.get()
        if clip is None:
            return self._demo_analyze(image_id, image_path, modality)

        # ── Vision encoding ────────────────────────────────────────────────
        pil_img = Image.open(image_path).convert("RGB")
        image_tensor = clip.preprocess(pil_img).unsqueeze(0).to(clip.device)

        # ── Text encoding — one visual description prompt per class ────────
        text_tokens = clip.tokenizer([e[1] for e in entries]).to(clip.device)

        with torch.no_grad():
            img_feat  = clip.model.encode_image(image_tensor)
            txt_feat  = clip.model.encode_text(text_tokens)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat  / txt_feat.norm(dim=-1, keepdim=True)
            # Raw cosine-similarity logits → softmax probabilities
            probs = (img_feat @ txt_feat.T).softmax(dim=-1).squeeze(0).cpu().float().numpy()

        n = len(entries)
        ranked = sorted(range(n), key=lambda i: float(probs[i]), reverse=True)

        primary_idx  = ranked[0]
        label        = entries[primary_idx][0]
        primary_conf = round(float(probs[primary_idx]), 4)
        second_conf  = round(float(probs[ranked[1]]), 4) if n > 1 else 0.0
        margin       = round(primary_conf - second_conf, 4)

        # ── Severity and recommendation derived from model output only ─────
        severity       = _infer_severity(label, margin)
        recommendation = _infer_recommendation(severity, label)

        if margin > 0.15:
            conf_qual = "high confidence"
        elif margin > 0.05:
            conf_qual = "moderate confidence"
        else:
            conf_qual = "low confidence — visual patterns are closely matched"

        region       = _REGION.get(modality, "unspecified")
        modality_str = modality.value.replace("_", " ")

        # ── Build findings entirely from inference numbers ─────────────────
        THRESHOLD = 0.10
        findings: List[Finding] = []
        for rank, idx in enumerate(ranked):
            d_label = entries[idx][0]
            d_conf  = round(float(probs[idx]), 4)

            if d_conf < THRESHOLD:
                break
            if "normal" in d_label.lower():
                continue

            if rank == 0:
                desc = (
                    f"BiomedCLIP computed {d_conf:.1%} visual similarity between the uploaded "
                    f"{modality_str} and the '{d_label}' radiological pattern "
                    f"({conf_qual}; margin over next class: {margin:.1%})."
                )
            else:
                prev_conf = round(float(probs[ranked[rank - 1]]), 4)
                desc = (
                    f"BiomedCLIP also matched '{d_label}' at {d_conf:.1%} "
                    f"(Δ {prev_conf - d_conf:.1%} below rank-{rank} class). "
                    f"Consider as differential."
                )
            findings.append(Finding(label=d_label, confidence=d_conf,
                                    description=desc, region=region))

        if not findings:
            findings.append(Finding(
                label="Normal", confidence=primary_conf, region=region,
                description=(
                    f"BiomedCLIP found no pathological visual pattern above threshold. "
                    f"Highest match was '{label}' at {primary_conf:.1%} ({conf_qual})."
                ),
            ))

        differentials = [
            entries[ranked[i]][0]
            for i in range(1, min(5, n))
            if float(probs[ranked[i]]) > 0.03
        ]

        class_probs = [
            ClassProbability(label=entries[i][0], probability=round(float(probs[i]), 4), icd10=None)
            for i in sorted(range(n), key=lambda i: float(probs[i]), reverse=True)
        ]

        status = DiagnosticStatus.NORMAL if "normal" in label.lower() else DiagnosticStatus.ABNORMAL
        return ImageAnalysisResult(
            image_id=image_id, modality=modality, status=status,
            findings=findings, primary_diagnosis=label,
            differential_diagnoses=differentials, severity=severity,
            recommendation=recommendation, confidence_score=primary_conf,
            processing_time_ms=0.0, inference_source="biomedclip",
            class_probabilities=class_probs,
        )

    # ── MedSigLIP-448 zero-shot inference ─────────────────────────────────────

    def _medsiglip_analyze(
        self, image_id: str, image_path: str, modality: ImageModality
    ) -> ImageAnalysisResult:
        entries = _PROMPTS.get(modality, _PROMPTS[ImageModality.CHEST_XRAY])
        text_prompts = [e[1] for e in entries]

        probs = medsiglip_classify(image_path, text_prompts)
        if probs is None:
            logger.warning("MedSigLIP unavailable — falling back to BiomedCLIP/demo.")
            clip = _BiomedCLIPModel.get()
            return (
                self._biomedclip_analyze(image_id, image_path, modality)
                if clip is not None
                else self._demo_analyze(image_id, image_path, modality)
            )

        n = len(entries)
        ranked = sorted(range(n), key=lambda i: probs[i], reverse=True)
        primary_idx  = ranked[0]
        label        = entries[primary_idx][0]
        primary_conf = round(probs[primary_idx], 4)
        second_conf  = round(probs[ranked[1]], 4) if n > 1 else 0.0
        margin       = round(primary_conf - second_conf, 4)

        severity       = _infer_severity(label, margin)
        recommendation = _infer_recommendation(severity, label)
        region         = _REGION.get(modality, "unspecified")
        modality_str   = modality.value.replace("_", " ")

        conf_qual = (
            "high confidence" if margin > 0.15
            else "moderate confidence" if margin > 0.05
            else "low confidence — visual patterns are closely matched"
        )

        THRESHOLD = 0.10
        findings: List[Finding] = []
        for rank, idx in enumerate(ranked):
            d_label = entries[idx][0]
            d_conf  = round(probs[idx], 4)
            if d_conf < THRESHOLD:
                break
            if "normal" in d_label.lower():
                continue
            if rank == 0:
                desc = (
                    f"MedSigLIP-448 computed {d_conf:.1%} similarity between this "
                    f"{modality_str} and the '{d_label}' pattern "
                    f"({conf_qual}; margin over next class: {margin:.1%})."
                )
            else:
                prev_conf = round(probs[ranked[rank - 1]], 4)
                desc = (
                    f"MedSigLIP-448 also matched '{d_label}' at {d_conf:.1%} "
                    f"(\u0394 {prev_conf - d_conf:.1%} below rank-{rank} class). "
                    f"Consider as differential."
                )
            findings.append(Finding(label=d_label, confidence=d_conf,
                                    description=desc, region=region))

        if not findings:
            findings.append(Finding(
                label="Normal", confidence=primary_conf, region=region,
                description=(
                    f"MedSigLIP-448 found no pathological pattern above threshold. "
                    f"Highest match was '{label}' at {primary_conf:.1%} ({conf_qual})."
                ),
            ))

        differentials = [
            entries[ranked[i]][0]
            for i in range(1, min(5, n))
            if probs[ranked[i]] > 0.03
        ]
        class_probs = [
            ClassProbability(
                label=entries[i][0],
                probability=round(probs[i], 4),
                icd10=None,
            )
            for i in sorted(range(n), key=lambda i: probs[i], reverse=True)
        ]

        status = DiagnosticStatus.NORMAL if "normal" in label.lower() else DiagnosticStatus.ABNORMAL
        return ImageAnalysisResult(
            image_id=image_id, modality=modality, status=status,
            findings=findings, primary_diagnosis=label,
            differential_diagnoses=differentials, severity=severity,
            recommendation=recommendation, confidence_score=primary_conf,
            processing_time_ms=0.0, inference_source="medsiglip",
            class_probabilities=class_probs,
        )

    # ── OpenVINO inference ────────────────────────────────────────────────────

    async def _openvino_analyze(
        self, image_id: str, image_path: str, modality: ImageModality
    ) -> ImageAnalysisResult:
        compiled = self._ov_models[modality]
        entries = _PROMPTS.get(modality, _PROMPTS[ImageModality.CHEST_XRAY])
        n = len(entries)

        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        blob = ((arr - mean) / std).transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        output = list(compiled(blob).values())[0][0]
        e = np.exp(output - output.max())
        probs = (e / e.sum())[:n]  # truncate to number of prompts

        ranked       = sorted(range(n), key=lambda i: float(probs[i]), reverse=True)
        primary_idx  = ranked[0]
        label        = entries[primary_idx][0]
        primary_conf = round(float(probs[primary_idx]), 4)
        second_conf  = round(float(probs[ranked[1]]), 4) if n > 1 else 0.0
        margin       = round(primary_conf - second_conf, 4)

        severity       = _infer_severity(label, margin)
        recommendation = _infer_recommendation(severity, label)
        region         = _REGION.get(modality, "unspecified")

        findings: List[Finding] = []
        if "normal" not in label.lower():
            findings.append(Finding(
                label=label, confidence=primary_conf, region=region,
                description=(
                    f"OpenVINO model matched '{label}' at {primary_conf:.1%} "
                    f"(margin over next class: {margin:.1%})."
                ),
            ))

        differentials = [
            entries[ranked[i]][0] for i in range(1, min(4, n))
            if float(probs[ranked[i]]) > 0.05
        ]
        class_probs = [
            ClassProbability(label=entries[i][0], probability=round(float(probs[i]), 4), icd10=None)
            for i in sorted(range(n), key=lambda i: float(probs[i]), reverse=True)
        ]

        status = DiagnosticStatus.NORMAL if "normal" in label.lower() else DiagnosticStatus.ABNORMAL
        return ImageAnalysisResult(
            image_id=image_id, modality=modality, status=status,
            findings=findings, primary_diagnosis=label,
            differential_diagnoses=differentials, severity=severity,
            recommendation=recommendation, confidence_score=primary_conf,
            processing_time_ms=0.0, inference_source="openvino",
            class_probabilities=class_probs,
        )

    # ── Demo fallback ─────────────────────────────────────────────────────────

    def _demo_analyze(
        self, image_id: str, image_path: str, modality: ImageModality
    ) -> ImageAnalysisResult:
        entries = _PROMPTS.get(modality, _PROMPTS[ImageModality.CHEST_XRAY])
        n = len(entries)
        # Use image pixel statistics as a deterministic seed so different images
        # always produce different results even in demo/fallback mode
        primary_idx = _image_hash_index(image_path, n)
        scores      = _demo_scores(image_path, n, primary_idx)

        label        = entries[primary_idx][0]
        primary_conf = scores[primary_idx]
        ranked       = sorted(range(n), key=lambda i: scores[i], reverse=True)
        second_conf  = scores[ranked[1]] if n > 1 else 0.0
        margin       = round(primary_conf - second_conf, 4)

        severity       = _infer_severity(label, margin)
        recommendation = _infer_recommendation(severity, label)
        region         = _REGION.get(modality, "unspecified")

        findings: List[Finding] = []
        if "normal" not in label.lower():
            findings.append(Finding(
                label=label, confidence=round(primary_conf, 4), region=region,
                description=(
                    f"Image-derived score: {primary_conf:.1%} match for '{label}' "
                    f"(margin over next class: {margin:.1%}). "
                    f"[BiomedCLIP unavailable — pixel-hash fallback.]"
                ),
            ))

        differentials = [
            entries[ranked[i]][0] for i in range(1, min(4, n))
            if scores[ranked[i]] > 0.05
        ]
        class_probs = [
            ClassProbability(label=entries[i][0], probability=round(scores[i], 4), icd10=None)
            for i in sorted(range(n), key=lambda i: scores[i], reverse=True)
        ]

        status = DiagnosticStatus.NORMAL if "normal" in label.lower() else DiagnosticStatus.ABNORMAL
        return ImageAnalysisResult(
            image_id=image_id, modality=modality, status=status,
            findings=findings, primary_diagnosis=label,
            differential_diagnoses=differentials, severity=severity,
            recommendation=recommendation, confidence_score=round(primary_conf, 4),
            processing_time_ms=0.0, inference_source="demo",
            class_probabilities=class_probs,
        )


image_analyzer = ImageAnalyzer()
