"""SAR raster as a PNG for the map.

A full XYZ tile pyramid would be the right answer for an operational GIS, and
the GeoTIFF on disk is already georeferenced so one can be generated. For a
1024x1024 scene the whole raster is a 1 MB PNG and a single image overlay is
both faster and simpler than 341 tile requests, so that is what this serves.
The GeoTIFF itself is exposed alongside it for anyone who wants to pull the
scene into their own GIS.
"""
from __future__ import annotations

import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from api.app.db import SessionLocal
from api.app.models.records import Scene as SceneRow

router = APIRouter()

_CACHE: dict[str, bytes] = {}


@router.get("/scenes/{scene_id}/raster.png")
def raster_png(scene_id: str, vmin: float = -22.0, vmax: float = -4.0) -> Response:
    import numpy as np
    from core.sar.preprocess import read_scene

    key = f"{scene_id}:{vmin}:{vmax}"
    if key in _CACHE:
        return Response(_CACHE[key], media_type="image/png")

    with SessionLocal() as session:
        row = session.get(SceneRow, scene_id)
        if row is None:
            raise HTTPException(404, f"No scene '{scene_id}'.")
        path = row.raster_path

    from PIL import Image

    raster = read_scene(path)
    db = raster.vv_db
    scaled = np.clip((db - vmin) / (vmax - vmin), 0.0, 1.0)
    grey = (scaled * 255.0).astype(np.uint8)
    alpha = np.where(raster.valid & np.isfinite(db), 255, 0).astype(np.uint8)
    rgba = np.dstack([grey, grey, grey, alpha])
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    if len(_CACHE) > 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = payload
    return Response(payload, media_type="image/png")


@router.get("/scenes/{scene_id}/scene.tif")
def raster_tif(scene_id: str) -> FileResponse:
    """The calibrated sigma0 GeoTIFF, for pulling into an existing GIS."""
    with SessionLocal() as session:
        row = session.get(SceneRow, scene_id)
        if row is None:
            raise HTTPException(404, f"No scene '{scene_id}'.")
        return FileResponse(
            row.raster_path,
            media_type="image/tiff",
            filename=f"avanta_{scene_id}.tif",
        )
