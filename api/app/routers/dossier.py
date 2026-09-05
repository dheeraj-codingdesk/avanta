from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from api.app.db import SessionLocal
from api.app.models.records import AttributionRunRow
from api.app.models.records import Scene as SceneRow
from api.app.schemas.requests import DossierRequest
from core.ais.gfw import GfwClient

router = APIRouter()

_CACHE: dict[str, Any] = {}


def _load(run_id: str, mmsi: str, observer: str) -> Any:
    from core.dossier.builder import render

    key = f"{run_id}:{mmsi}"
    if key in _CACHE:
        return _CACHE[key]
    with SessionLocal() as session:
        run_row = session.get(AttributionRunRow, run_id)
        if run_row is None:
            raise HTTPException(404, f"No attribution run '{run_id}'.")
        scene_row = session.get(SceneRow, run_row.scene_id)
        scene = {
            "bbox": scene_row.bbox if scene_row else None,
            "acquired_utc": scene_row.acquired_utc if scene_row else None,
            "product_id": scene_row.product_id if scene_row else None,
            "raster_sha256": scene_row.raster_sha256 if scene_row else None,
        }
        result = dict(run_row.result)
        provenance = run_row.provenance or {}

    # Enrich the identity fields from the GFW registry where possible: an MMSI
    # alone is not enough for a flag State referral.
    client = GfwClient()
    if client.configured and not mmsi.startswith("DARK-"):
        identity = client.identity_for_mmsi(mmsi)
        if identity is not None:
            tracks = result.setdefault("tracks", {}).setdefault(mmsi, {"features": []})
            for feature in tracks.get("features", []):
                props = feature.setdefault("properties", {})
                if props.get("segment") == "transmitted":
                    props.setdefault("name", identity.name)
                    props["imo"] = props.get("imo") or identity.imo
                    props["flag"] = props.get("flag") or identity.flag
                    props["length_m"] = props.get("length_m") or identity.length_m
            provenance = {**provenance, "identity": identity.to_dict()}

    dossier = render(
        scene=scene,
        run=result,
        provenance=provenance,
        mmsi=mmsi,
        run_id=run_id,
        observer=observer,
    )
    _CACHE[key] = dossier
    if len(_CACHE) > 16:
        _CACHE.pop(next(iter(_CACHE)))
    return dossier


@router.post("/dossier/generate")
def generate(request: DossierRequest) -> dict[str, Any]:
    dossier = _load(request.run_id, request.mmsi, request.observer)
    return {
        "run_id": request.run_id,
        "mmsi": request.mmsi,
        "pdf_url": f"/api/v1/dossier/{request.run_id}/{request.mmsi}/pdf",
        "json_url": f"/api/v1/dossier/{request.run_id}/{request.mmsi}/json",
        "html_url": f"/api/v1/dossier/{request.run_id}/{request.mmsi}/html",
        "manifest_sha256": dossier.manifest["manifest_sha256"],
        "fields": dossier.fields,
    }


@router.get("/dossier/{run_id}/{mmsi}/json")
def as_json(run_id: str, mmsi: str) -> dict[str, Any]:
    return _load(run_id, mmsi, "AVANTA automated analysis").to_json()


@router.get("/dossier/{run_id}/{mmsi}/html")
def as_html(run_id: str, mmsi: str) -> HTMLResponse:
    return HTMLResponse(_load(run_id, mmsi, "AVANTA automated analysis").html)


@router.get("/dossier/{run_id}/{mmsi}/pdf")
def as_pdf(run_id: str, mmsi: str) -> FileResponse:
    dossier = _load(run_id, mmsi, "AVANTA automated analysis")
    if dossier.pdf_path is None or not dossier.pdf_path.exists():
        raise HTTPException(500, "PDF generation did not produce a file.")
    return FileResponse(
        dossier.pdf_path,
        media_type="application/pdf",
        filename=f"AVANTA_MARPOL_Appendix3_{mmsi}.pdf",
    )
