# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
VLM Medical Image Analyzer  —  MedGemma + MedSigLIP
=====================================================
Uses google/medgemma-4b-it — Google's multimodal medical foundation model.

Architecture:
  • Vision encoder : MedSigLIP  (SigLIP trained on medical image–text pairs)
  • Language model : Gemma 3 4B instruction-tuned
  • Combined via   : PaliGemma-style late fusion

The model generates a structured clinical report covering:
  • Radiological / visual findings
  • Primary diagnosis with reasoning
  • Severity assessment
  • Differential diagnoses
  • Treatment approach and recommendations

Model is loaded once (singleton) using bfloat16 on GPU, float32 on CPU.
Requires a Hugging Face token with access to the gated google/medgemma-4b-it
repository.  Token is read from HF_TOKEN env var or ~/.cache/huggingface/token.
Falls back gracefully when the model cannot be loaded.
"""

import logging
import os
import threading
import re
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_VLM_MODEL_ID = "google/medgemma-4b-it"


def _get_hf_token() -> Optional[str]:
    """Return HF token from env or credential file."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_file):
        return open(token_file).read().strip()
    return None


# ─────────────────────────── Singleton model holder ──────────────────────────

class _MedGemmaModel:
    _instance: Optional["_MedGemmaModel"] = None
    _failed: bool = False
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        token = _get_hf_token()
        if not token:
            raise RuntimeError(
                "No Hugging Face token found. Set HF_TOKEN env var or run "
                "`huggingface-cli login` to access google/medgemma-4b-it."
            )

        logger.info(
            "Loading MedGemma: %s (MedSigLIP vision encoder + Gemma 3 4B LM) "
            "— this may take 1–3 minutes on first call", _VLM_MODEL_ID
        )

        self.processor = AutoProcessor.from_pretrained(
            _VLM_MODEL_ID, token=token
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForImageTextToText.from_pretrained(
            _VLM_MODEL_ID,
            token=token,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(
            "MedGemma loaded on %s (dtype=%s)", self.device, dtype
        )

    @classmethod
    def get(cls) -> Optional["_MedGemmaModel"]:
        if cls._instance is not None:
            return cls._instance
        if cls._failed:
            return None
        with cls._lock:
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
                    "MedGemma load failed (%s) — VLM report unavailable.", exc
                )
                return None


# ─────────────────────────── Prompt builder ───────────────────────────────────

_MODALITY_CONTEXT = {
    "chest_xray": "chest X-ray (PA/AP projection)",
    "ct_scan":    "CT scan",
    "mri":        "MRI scan",
    "dermatology":"dermatological / skin lesion image",
    "pathology":  "histopathology / microscopy slide",
    "other":      "medical image",
}


def _build_messages(modality: str, clinical_notes: Optional[str], pil_image: Image.Image) -> list:
    """Build MedGemma chat messages with embedded image."""
    modality_name = _MODALITY_CONTEXT.get(modality, "medical image")
    notes_line = (
        f"\nClinical context: {clinical_notes.strip()}" if clinical_notes else ""
    )
    prompt = (
        f"You are an expert radiologist and clinician. "
        f"Analyze this {modality_name}.{notes_line}\n\n"
        f"Provide a structured clinical report with exactly these five sections:\n\n"
        f"FINDINGS\n"
        f"Describe all radiological or visual findings in detail. Use precise medical "
        f"terminology (opacity, consolidation, lesion margins, distribution, density, etc.).\n\n"
        f"PRIMARY DIAGNOSIS\n"
        f"State the most likely diagnosis. Explain which imaging features support it.\n\n"
        f"SEVERITY\n"
        f"Rate severity as one of: Normal / Mild / Moderate / Severe / Critical. "
        f"Justify with specific findings.\n\n"
        f"DIFFERENTIAL DIAGNOSES\n"
        f"List 3–5 alternative diagnoses with brief reasoning.\n\n"
        f"TREATMENT & RECOMMENDATIONS\n"
        f"1. Immediate actions (if urgent)\n"
        f"2. Recommended investigations or follow-up imaging\n"
        f"3. Treatment approach (pharmacological, surgical, or supportive)\n"
        f"4. Monitoring and patient counselling\n"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text":  prompt},
            ],
        }
    ]


# ─────────────────────────── Section parser ──────────────────────────────────

def _parse_sections(raw_text: str) -> dict:
    """Extract the five named sections from the LLaVA response."""
    sections = {
        "findings": "",
        "primary_diagnosis": "",
        "severity": "",
        "differential_diagnoses": "",
        "treatment": "",
        "raw": raw_text.strip(),
    }
    # Strip anything before the first section header
    patterns = [
        ("findings",              r"\*{0,2}FINDINGS?\*{0,2}"),
        ("primary_diagnosis",     r"\*{0,2}PRIMARY\s+DIAGNOSIS?\*{0,2}"),
        ("severity",              r"\*{0,2}SEVERITY\*{0,2}"),
        ("differential_diagnoses",r"\*{0,2}DIFFERENTIAL\s+DIAGNOS(?:IS|ES)?\*{0,2}"),
        ("treatment",             r"\*{0,2}TREATMENT\s*(?:&|AND)?\s*RECOMMENDATIONS?\*{0,2}"),
    ]
    compiled = [(k, re.compile(p, re.IGNORECASE)) for k, p in patterns]

    # Find positions
    positions = []
    for key, rex in compiled:
        m = rex.search(raw_text)
        if m:
            positions.append((m.start(), m.end(), key))
    positions.sort()

    for i, (start, end, key) in enumerate(positions):
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(raw_text)
        sections[key] = raw_text[end:next_start].strip(" :\n")

    return sections


# ─────────────────────────── Public interface ─────────────────────────────────

class VLMReport:
    """Structured report returned by the VLM analyzer."""
    __slots__ = (
        "findings", "primary_diagnosis", "severity",
        "differential_diagnoses", "treatment", "raw", "model_id",
    )

    def __init__(
        self,
        findings: str,
        primary_diagnosis: str,
        severity: str,
        differential_diagnoses: str,
        treatment: str,
        raw: str,
        model_id: str,
    ):
        self.findings               = findings
        self.primary_diagnosis      = primary_diagnosis
        self.severity               = severity
        self.differential_diagnoses = differential_diagnoses
        self.treatment              = treatment
        self.raw                    = raw
        self.model_id               = model_id

    def to_dict(self) -> dict:
        return {
            "findings":               self.findings,
            "primary_diagnosis":      self.primary_diagnosis,
            "severity":               self.severity,
            "differential_diagnoses": self.differential_diagnoses,
            "treatment":              self.treatment,
            "raw":                    self.raw,
            "model_id":               self.model_id,
        }


def analyze_image_with_vlm(
    image_path: str,
    modality: str,
    clinical_notes: Optional[str] = None,
    max_new_tokens: int = 600,
) -> Optional[VLMReport]:
    """
    Run MedGemma (MedSigLIP vision encoder + Gemma 3 4B) on a medical image
    and return a structured VLMReport.  Returns None if the model is unavailable.
    """
    vlm = _MedGemmaModel.get()
    if vlm is None:
        return None

    import torch

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error("Cannot open image %s: %s", image_path, exc)
        return None

    messages = _build_messages(modality, clinical_notes, pil_image)

    try:
        # Apply Gemma chat template — tokenize=True returns a BatchEncoding dict
        inputs = vlm.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(vlm.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = vlm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][input_len:]
        raw_output = vlm.processor.decode(new_tokens, skip_special_tokens=True).strip()

        sections = _parse_sections(raw_output)
        return VLMReport(
            findings=sections["findings"],
            primary_diagnosis=sections["primary_diagnosis"],
            severity=sections["severity"],
            differential_diagnoses=sections["differential_diagnoses"],
            treatment=sections["treatment"],
            raw=sections["raw"],
            model_id=_VLM_MODEL_ID,
        )

    except Exception as exc:
        logger.error("MedGemma inference error: %s", exc, exc_info=True)
        return None
