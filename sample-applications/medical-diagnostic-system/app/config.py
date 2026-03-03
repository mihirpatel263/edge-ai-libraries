# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    APP_TITLE: str = "Intelligent Multimodal Medical Diagnostic System"
    APP_VERSION: str = "1.0.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8080
    ROOT_PATH: str = "/v1/meddiag"

    # CORS
    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./medical_diagnostic.db"

    # Storage for uploaded images
    UPLOAD_DIR: str = "./uploads"
    MAX_IMAGE_SIZE_MB: int = 50

    # LLM for report generation (optional)
    LLM_ENDPOINT_URL: Optional[str] = None
    LLM_MODEL: str = "gpt-4"
    LLM_MAX_TOKENS: int = 1024

    # OpenVINO model paths (optional, uses demo mode when not set)
    OPENVINO_CHEST_XRAY_MODEL: Optional[str] = None
    OPENVINO_CT_MODEL: Optional[str] = None
    OPENVINO_MRI_MODEL: Optional[str] = None

    # Demo mode — uses simulated analysis when True
    DEMO_MODE: bool = True

    # OpenTelemetry
    OTLP_ENDPOINT: Optional[str] = None
    OTEL_SERVICE_NAME: str = "medical-diagnostic-system"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
