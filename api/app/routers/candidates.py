from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api.app import jobs, state
from api.app.db import SessionLocal
from api.app.models.records import CandidateSet
from api.app.models.records import Scene as SceneRow
from api.app.schemas.requests import CandidatesRequest
from core import scenarios as scenario_module
from core.config import REPO_ROOT
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.pipeline import SceneBundle
from core.provenance.record import SourceRecord

router = APIRouter()


def resolve_tracks(bundle: SceneBundle, scenario_id: str | None, source: str) -> tuple[list, SourceRecord]:
    """Live AIS if it has anything useful, otherwise the scenario's fixture.

    Which one was used is returned, always, so the UI can badge it.
    """
    # A generated scenario carries its own fleet. It exists nowhere else, and
    # substituting live traffic for it would break the ground truth the whole
    # case is built on.
    generated = state.get_tracks(bundle.scene.scene_id)
    if generated:
        return generated, SourceRecord(
            source="AVANTA synthetic AIS fleet (generated)",
            mode="SYNTHETIC",
            detail={"n_vessels": len(generated), "scene": bundle.scene.scene_id},
        )

    if source != "fixture":
        live = state.collector.tracks()
        in_box = [
            t
            for t in live
            if any(
                bundle.scene.bbox[0] <= f.lon <= bundle.scene.bbox[2]
                and bundle.scene.bbox[1] <= f.lat <= bundle.scene.bbox[3]
                for f in t.fixes
            )
        ]
        if in_box:
            return in_box, SourceRecord(
                source="aisstream.io live stream",
                mode="LIVE",
                detail={"n_vessels": len(in_box), **state.collector.status()},
            )
        if source == "live":
            raise HTTPException(
                503,
                "Live AIS is requested but no vessels have been observed in this "
                "bounding box yet. The collector needs time to accumulate a track.",
            )

    fixture_path: Path | None = None
    if scenario_id:
        scenario = scenario_module.get(scenario_id)
        if scenario and scenario.raw.get("ais_fixture"):
            candidate = REPO_ROOT / scenario.raw["ais_fixture"]
            if candidate.exists():
                fixture_path = candidate
    if fixture_path is None:
        available = sorted((REPO_ROOT / "fixtures" / "ais").glob("*.json"))
        fixture_path = available[0] if available else None
    if fixture_path is None:
        raise HTTPException(
            503,
            "No live AIS in this box and no bundled AIS fixture. "
            "Run scripts/capture_ais.py to record one.",
        )
    from core.pipeline import load_ais_fixture

    return load_ais_fixture(fixture_path, bundle.scene.bbox)


@router.post("/candidates/generate")
def generate(request: CandidatesRequest) -> dict[str, Any]:
    bundle = state.get_bundle(request.scene_id)
    if bundle is None:
        raise HTTPException(
            409,
            f"Scene '{request.scene_id}' is not loaded in this process. Re-run the "
            "ingest for this scene.",
        )
    with SessionLocal() as session:
        row = session.get(SceneRow, request.scene_id)
        scenario_id = row.scenario if row else None

    def work(handle: jobs.JobHandle) -> dict[str, Any]:
        from core.pipeline import generate_candidates

        handle.update(stage="resolving AIS source", progress=0.1)
        tracks, record = resolve_tracks(bundle, scenario_id, request.source)
        handle.update(
            stage=f"building tracks for {len(tracks)} vessels",
            progress=0.4,
            log_line=f"AIS source: {record.source} ({record.mode})",
        )
        result = generate_candidates(bundle, tracks, keep_top_k=request.keep_top_k)
        handle.update(stage="geometric prefilter", progress=0.85)

        set_id = uuid.uuid4().hex[:16]
        provenance = bundle.provenance.to_dict()
        provenance["ais"] = record.to_dict()
        with SessionLocal() as session:
            session.add(
                CandidateSet(
                    id=set_id,
                    scene_id=request.scene_id,
                    n_considered=result["n_considered"],
                    n_kept=result["n_kept"],
                    results=result["results"],
                    tracks=result["tracks"],
                    provenance=provenance,
                )
            )
            session.commit()
        return {
            "candidate_set_id": set_id,
            "scene_id": request.scene_id,
            "n_considered": result["n_considered"],
            "n_kept": result["n_kept"],
            "n_dark": result.get("n_dark", 0),
            "ais_mode": record.mode,
        }

    return {"job_id": jobs.submit("candidates", work)}


@router.get("/candidates")
def list_candidates(scene_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    with SessionLocal() as session:
        row = session.scalars(
            select(CandidateSet)
            .where(CandidateSet.scene_id == scene_id)
            .order_by(CandidateSet.created_at.desc())
            .limit(1)
        ).first()
        if row is None:
            raise HTTPException(404, f"No candidate set has been generated for scene '{scene_id}'.")
        return {
            "candidate_set_id": row.id,
            "scene_id": row.scene_id,
            "n_considered": row.n_considered,
            "n_kept": row.n_kept,
            "results": row.results,
            "tracks": row.tracks,
            "provenance": row.provenance,
        }
