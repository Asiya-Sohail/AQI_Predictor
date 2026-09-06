"""
hopsworks_client.py — Shared Hopsworks authentication utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict

try:
    import hopsworks
except Exception:  # pragma: no cover - optional in lightweight deploys
    hopsworks = None

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT,
    WINDOWS_TMP_DIR,
)


def _prepare_windows_temp_dir() -> None:
    """Ensure TMP/TEMP paths exist on Windows for certificate/temp-file handling."""
    if os.name != "nt":
        return

    tmp_dir = Path(WINDOWS_TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = str(tmp_dir)
    os.environ.setdefault("TMP", tmp_path)
    os.environ.setdefault("TEMP", tmp_path)
    os.environ.setdefault("TMPDIR", tmp_path)


def login_to_hopsworks() -> Any:
    """Authenticate and return the current Hopsworks project handle."""
    if hopsworks is None:
        raise RuntimeError(
            "hopsworks package is not installed. "
            "Install it to use Hopsworks registry/feature-store paths."
        )

    _prepare_windows_temp_dir()

    login_kwargs: Dict[str, str] = {
        "api_key_value": HOPSWORKS_API_KEY,
        "project": HOPSWORKS_PROJECT,
    }
    if HOPSWORKS_HOST.strip():
        login_kwargs["host"] = HOPSWORKS_HOST.strip()

    return hopsworks.login(**login_kwargs)
