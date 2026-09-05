from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from api.app import jobs, state
from api.app.db import SessionLocal
from api.app.models.records import AttributionRunRow, CandidateSet
from api.app.models.records import Scene as SceneRow
from api.app.routers.candidates import resolve_tracks
from api.app.schemas.requests import AttributionRequest

router = APIRouter()


@router.post("/attribution/run")
def run(request: AttributionRequest) -> dict[str, Any]:
    bundle = state.get_bundle(request.scene_id)
    if bundle is None:
        raise HTTPException(409, f"Scene '{request.scene_id}' is not loaded. Re-run the ingest.")
    if not bundle.detection.slicks():
        raise HTTPException(
            409,
            "No slick was segmented in this scene, so there is nothing to attribute. "
            "This is a valid result, not a failure.",
        )
    if not bundle.wind_gate.passed:
        raise HTTPException(
            409,
            f"Scene is wind-gated: {bundle.wind_gate.verdict} Attribution on a gated "
            "scene would rest on a detection the physics does not support.",
        )

    with SessionLocal() as session:
        scene_row = session.get(SceneRow, request.scene_id)
        scenario_id = scene_row.scenario if scene_row else None
        candidate_row = session.scalars(
            select(CandidateSet)
            .where(CandidateSet.scene_id == request.scene_id)
            .order_by(CandidateSet.created_at.desc())
            .limit(1)
        ).first()
        candidate_set_id = candidate_row.id if candidate_row else None
        kept_ids = (
            [r["mmsi"] for r in candidate_row.results if r["kept"]] if candidate_row else None
        )

    wanted = request.candidate_ids or kept_ids

    def work(handle: jobs.JobHandle) -> dict[str, Any]:
        from core.pipeline import attribute

        handle.update(stage="resolving candidate tracks", progress=0.02)
        tracks, record = resolve_tracks(bundle, scenario_id, "auto")
        dark_tracks = []
        candidates_result = None
        if wanted:
            by_mmsi = {t.mmsi: t for t in tracks}
            selected = [by_mmsi[m] for m in wanted if m in by_mmsi]
            missing = [m for m in wanted if m not in by_mmsi]
            if missing:
                # Dark contacts are not in the AIS set by definition; rebuild them.
                from core.pipeline import generate_candidates

                candidates_result = generate_candidates(bundle, tracks)
                from core.ais.darkmatch import DarkContact
                from core.ais.tracks import utc

                for contact in candidates_result.get("dark_contacts", []):
                    track = DarkContact(
                        contact_id=contact["contact_id"],
                        lon=contact["lon"],
                        lat=contact["lat"],
                        acquired_utc=utc(contact["acquired_utc"]),
                        nearest_ais_km=contact["nearest_ais_km"],
                        nearest_mmsi=contact["nearest_mmsi"],
                    ).as_track()
                    if track.mmsi in missing:
                        dark_tracks.append(track)
            tracks = selected + dark_tracks
        if not tracks:
            raise RuntimeError(
                "No candidate tracks survived selection. Generate candidates for this scene first."
            )

        handle.update(stage=f"simulating {len(tracks)} candidates", progress=0.05)

        def progress(stage: str, fraction: float) -> None:
            handle.update(stage=stage, progress=0.05 + 0.9 * fraction, log_line=stage)

        result = attribute(
            bundle,
            tracks,
            n_ensemble=request.n_ensemble,
            n_per_point=request.n_per_point,
            oil_type=request.oil_type,
            progress=progress,
        )
        handle.update(stage="scoring posterior", progress=0.96)

        run_id = uuid.uuid4().hex[:16]
        simulations = {
            c.mmsi: c.best_simulation.to_timeseries_geojson() for c in result.candidates
        }
        payload = result.to_dict()
        payload["evidence"] = {c.mmsi: c.evidence_breakdown() for c in result.candidates}
        payload["tracks"] = {t.mmsi: t.to_geojson() for t in tracks}
        payload["slick"] = bundle.detection.to_geojson()
        payload["wind_gate"] = bundle.wind_gate.to_dict()

        provenance = bundle.provenance.to_dict()
        provenance["ais"] = record.to_dict()
        with SessionLocal() as session:
            session.add(
                AttributionRunRow(
                    id=run_id,
                    scene_id=request.scene_id,
                    candidate_set_id=candidate_set_id,
                    result=payload,
                    simulations=simulations,
                    provenance=provenance,
                    runtime_s=result.runtime_s,
                )
            )
            scene = session.get(SceneRow, request.scene_id)
            if scene is not None:
                scene.status = "ATTRIBUTED" if not result.posterior.no_attribution else "NO_ATTRIBUTION"
            session.commit()
        top = result.posterior.top()
        return {
            "run_id": run_id,
            "scene_id": request.scene_id,
            "n_candidates": len(result.candidates),
            "p_null": round(result.posterior.p_null, 4),
            "no_attribution": result.posterior.no_attribution,
            "top": top.to_dict() if top else None,
            "runtime_s": round(result.runtime_s, 2),
        }

    return {"job_id": jobs.submit("attribution", work)}


@router.get("/attribution/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        row = session.get(AttributionRunRow, run_id)
        if row is None:
            raise HTTPException(404, f"No attribution run '{run_id}'.")
        return {
            "run_id": row.id,
            "scene_id": row.scene_id,
            "candidate_set_id": row.candidate_set_id,
            "runtime_s": row.runtime_s,
            "provenance": row.provenance,
            **row.result,
        }


@router.get("/attribution/{run_id}/sim/{mmsi}")
def get_simulation(run_id: str, mmsi: str) -> dict[str, Any]:
    """Particle positions per output step, for the timeline scrubber."""
    with SessionLocal() as session:
        row = session.get(AttributionRunRow, run_id)
        if row is None:
            raise HTTPException(404, f"No attribution run '{run_id}'.")
        sim = (row.simulations or {}).get(mmsi)
        if sim is None:
            raise HTTPException(404, f"No simulation for '{mmsi}' in run '{run_id}'.")
        return sim
