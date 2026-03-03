# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
MedSigLIP-448 Medical Image Analyser
======================================
Uses google/medsiglip-448 — Google's SigLIP vision–language model trained
specifically on medical image–text pairs at 448×448 resolution.

Architecture:
  • Vision encoder : ViT-L/14 at 448×448 px (4× the BiomedCLIP resolution)
  • Text encoder   : Transformer-based (same SigLIP architecture)
  • Training       : Contrastive medical image–text pairs from PubMed/medical datasets

Zero-shot classification:
  1. Encode the medical image with the vision encoder
  2. Encode per-class radiological text descriptions with the text encoder
  3. Compute cosine similarities → softmax probabilities
  4. Return ranked disease probabilities

Public API mirrors the BiomedCLIP interface in image_analyzer.py.
"""

import logging
import threading
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

MEDSIGLIP_MODEL_ID = "google/medsiglip-448"


# ─────────────────────────── Singleton loader ────────────────────────────────

class _MedSigLIPModel:
    _instance: Optional["_MedSigLIPModel"] = None
    _failed: bool = False
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        import torch
        from transformers import SiglipProcessor, SiglipModel

        logger.info("Loading MedSigLIP: %s (448×448 medical vision encoder)", MEDSIGLIP_MODEL_ID)
        self.processor = SiglipProcessor.from_pretrained(MEDSIGLIP_MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = SiglipModel.from_pretrained(
            MEDSIGLIP_MODEL_ID,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("MedSigLIP loaded on %s (dtype=%s)", self.device, dtype)

    @classmethod
    def get(cls) -> Optional["_MedSigLIPModel"]:
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
                logger.warning("MedSigLIP load failed (%s) — falling back to BiomedCLIP.", exc)
                return None


def is_medsiglip_available() -> bool:
    return _MedSigLIPModel.get() is not None


def medsiglip_classify(
    image_path: str,
    text_prompts: List[str],
) -> Optional[List[float]]:
    """
    Run MedSigLIP-448 zero-shot classification on a medical image.

    Args:
        image_path   : Path to the image file.
        text_prompts : List of radiological text descriptions (one per class).

    Returns:
        List of softmax probabilities (same length as text_prompts),
        or None if the model is unavailable.
    """
    model = _MedSigLIPModel.get()
    if model is None:
        return None

    import torch

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error("Cannot open image %s: %s", image_path, exc)
        return None

    try:
        inputs = model.processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.model(**inputs)
            # logits_per_image: (1, num_texts)
            logits = outputs.logits_per_image.squeeze(0)  # (num_texts,)
            # Use softmax for a proper probability distribution
            probs = logits.float().softmax(dim=-1).cpu().numpy()

        return [round(float(p), 4) for p in probs]

    except Exception as exc:
        logger.error("MedSigLIP inference error: %s", exc, exc_info=True)
        return None
