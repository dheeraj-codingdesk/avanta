from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from api.app import jobs, state
from api.app.db import SessionLocal
from api.app.models.records import Scene as SceneRow
from api.app.schemas.requests import IngestRequest
from core import scenarios as scenario_module
from core.sar.ingest import CdseClient

router = APIRouter()


@router.get("/scenarios")
def list_scenarios() -> dict[str, Any]:
    return {"scenarios": [s.to_dict() for s in scenario_module.load_all().values()]}


@router.post("/scenes/search")
def search(payload: dict[str, Any]) -> dict[str, Any]:
    bbox = payload["bbox"]
    t_from = payload["t_from"]
    t_to = payload["t_to"]
    client = CdseClient()
    if not client.configured:
        raise HTTPException(503, "CDSE credentials are not configured on this deployment.")
    features = client.search(bbox, t_from, t_to)
    return {
        "acquisitions": [
            {
                "id": f["id"],
                "datetime": (f.get("properties") or {}).get("datetime"),
                "mode": (f.get("properties") or {}).get("sar:instrument_mode"),
                "geometry": f.get("geometry"),
            }
            for f in features
        ],
        "count": len(features),
    }


def _window_for(request: IngestRequest) -> tuple[list[float], str, str, str | None]:
    if request.scenario:
        scenario = scenario_module.get(request.scenario)
        if scenario is None:
            raise HTTPException(404, f"Unknown scenario '{request.scenario}'.")
        bbox = scenario.bbox
        if scenario.kind == "live":
            days = int(scenario.raw.get("lookback_days", 6))
            now = datetime.now(timezone.utc)
            return bbox, (now - timedelta(days=days)).isoformat(), now.isoformat(), scenario.id
        return bbox, scenario.raw["t_from"], scenario.raw["t_to"], scenario.id
    if not (request.bbox and request.t_from and request.t_to):
        raise HTTPException(422, "Provide either a scenario id, or bbox with t_from and t_to.")
    return request.bbox, request.t_from, request.t_to, None


@router.post("/scenes/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    bbox, t_from, t_to, scenario_id = _window_for(request)

    scenario = scenario_module.get(scenario_id) if scenario_id else None
    is_synthetic = bool(scenario and scenario.kind == "synthetic")

    def work(handle: jobs.JobHandle) -> dict[str, Any]:
        from core.pipeline import ingest_scene, ingest_synthetic

        def progress(stage: str, fraction: float) -> None:
            handle.update(stage=stage, progress=fraction, log_line=stage)

        ground_truth: dict[str, Any] | None = None
        if is_synthetic:
            bundle, synthetic_tracks, ground_truth = ingest_synthetic(progress=progress)
            state.put_tracks("synthetic-discharge", synthetic_tracks)
        else:
            bundle = ingest_scene(
                bbox, t_from, t_to, allow_live=request.allow_live, progress=progress
            )
        scene_id = (
            "synthetic-discharge"
            if is_synthetic
            else hashlib.sha256(f"{bbox}{t_from}{t_to}{bundle.scene.sha256}".encode()).hexdigest()[:16]
        )
        state.put_bundle(scene_id, bundle)

        detections = bundle.detection.to_geojson()
        row = SceneRow(
            id=scene_id,
            scenario=scenario_id,
            bbox=bundle.scene.bbox,
            t_from=bundle.scene.t_from,
            t_to=bundle.scene.t_to,
            acquired_utc=bundle.acquired_utc.isoformat(),
            product_id=bundle.scene.product_id,
            raster_path=str(bundle.scene.path),
            raster_sha256=bundle.scene.sha256,
            mode=bundle.mode,
            status="GATED" if not bundle.wind_gate.passed else "NEW",
            wind_gate={**bundle.wind_gate.to_dict(), "coverage": bundle.coverage.to_dict()},
            detections=detections,
            provenance=bundle.provenance.to_dict(),
            currents_path=str(bundle.currents_path),
            wind_path=str(bundle.wind_path),
        )
        with SessionLocal() as session:
            session.merge(row)
            session.commit()
        return {
            "scene_id": scene_id,
            "mode": bundle.mode,
            "wind_gate": bundle.wind_gate.to_dict(),
            "coverage": bundle.coverage.to_dict(),
            "n_regions": len(bundle.detection.regions),
            "n_slicks": len(bundle.detection.slicks()),
            "ground_truth": ground_truth,
        }

    return {"job_id": jobs.submit("scene_ingest", work)}


@router.get("/scenes")
def list_scenes(limit: int = 50) -> dict[str, Any]:
    with SessionLocal() as session:
        rows = session.scalars(
            select(SceneRow).order_by(SceneRow.created_at.desc()).limit(limit)
        ).all()
        return {
            "scenes": [
                {
                    "id": r.id,
                    "scenario": r.scenario,
                    "bbox": r.bbox,
                    "acquired_utc": r.acquired_utc,
                    "mode": r.mode,
                    "status": r.status,
                    "wind_gate": r.wind_gate,
                    "n_slicks": len(
                        [
                            f
                            for f in ((r.detections or {}).get("features") or [])
                            if (f.get("properties") or {}).get("class") == "oil"
                        ]
                    ),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        }


@router.get("/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        row = session.get(SceneRow, scene_id)
        if row is None:
            raise HTTPException(404, f"No scene '{scene_id}'.")
        return {
            "id": row.id,
            "scenario": row.scenario,
            "bbox": row.bbox,
            "t_from": row.t_from,
            "t_to": row.t_to,
            "acquired_utc": row.acquired_utc,
            "product_id": row.product_id,
            "mode": row.mode,
            "status": row.status,
            "wind_gate": row.wind_gate,
            "provenance": row.provenance,
            "tile_url": f"/api/v1/scenes/{row.id}/raster.png",
        }


@router.get("/scenes/{scene_id}/detections")
def get_detections(scene_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        row = session.get(SceneRow, scene_id)
        if row is None:
            raise HTTPException(404, f"No scene '{scene_id}'.")
        bundle = state.get_bundle(scene_id)
        ships = bundle.detection.ship_pixels if bundle else []
        return {
            "detections": row.detections,
            "ship_contacts": [{"lon": lon, "lat": lat} for lon, lat in ships],
            "wind_gate": row.wind_gate,
            "coverage": (row.wind_gate or {}).get("coverage"),
            "mode": row.mode,
            "provenance": row.provenance,
        }
