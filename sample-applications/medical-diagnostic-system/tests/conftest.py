# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Top-level conftest — sets environment variables BEFORE any app modules are
imported so that Pydantic Settings picks them up correctly.
"""

import os
import pytest

# ─── Set test environment before any app module is imported ───────────────────
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_medical.db")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("UPLOAD_DIR", "/tmp/meddiag_test_uploads")
os.makedirs("/tmp/meddiag_test_uploads", exist_ok=True)
