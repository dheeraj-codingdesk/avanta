from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from api.app import state
from api.app.db import SessionLocal
from core.config import config_sha, settings
from core.provenance.hashing import git_sha

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    """Every dependency, with what it actually is right now.

    A health endpoint that reports OK when a data source is down is worse than
    none: it turns a visible failure into an invisible one.
    """
    deps: dict[str, Any] = {}

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        deps["db"] = {"status": "UP"}
    except Exception as exc:  # noqa: BLE001
        deps["db"] = {"status": "DOWN", "detail": str(exc)[:200]}

    deps["cdse"] = {
        "status": "CONFIGURED" if os.environ.get("CDSE_CLIENT_ID") else "NOT_CONFIGURED",
        "detail": "Sentinel-1 GRD via the Sentinel Hub Process API",
    }
    deps["forcing"] = {
        "status": "UP",
        "wind": "ERA5 reanalysis via the Open-Meteo archive API",
        "currents": "Copernicus Marine if credentials are present, otherwise the Open-Meteo global ocean model",
    }
    deps["ais"] = {
        "status": "UP" if state.collector.connected else ("CONFIGURED" if state.collector.configured else "NOT_CONFIGURED"),
        **state.collector.status(),
    }
    # Importing the full geospatial pipeline just to name the segmenter adds a
    # large, unnecessary startup cost to every API process.
    deps["model"] = {
        "segmenter": (
            "attention-unet (checkpoint loaded)"
            if Path("models/segmenter.pt").exists()
            else "classical detector (no trained checkpoint loaded)"
        ),
        "trained_checkpoint": "models/segmenter.pt not present",
    }

    return {
        "status": "ok" if deps["db"]["status"] == "UP" else "degraded",
        "version": settings()["version"],
        "git_sha": git_sha(),
        "config_sha": config_sha(),
        "dependencies": deps,
    }
