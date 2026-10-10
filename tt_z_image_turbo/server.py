# SPDX-License-Identifier: Apache-2.0
"""HTTP server for Z-Image-Turbo on one Blackhole p100a.

    uvicorn tt_z_image_turbo.server:app --host 0.0.0.0 --port 20000

One model, one card, one process. Requests are serialised: the card runs one image at a time.
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import HF_REPO, HF_REVISION, SCHED
from .pipeline import ZImageTurboTT, close_device, dram_stats, open_device

LICENSE = "apache-2.0"
MIN_DIM, MAX_DIM, DIM_STEP, MAX_PIXELS = 512, 1920, 8, 1920 * 1088
TURN_WAIT_S = float(os.environ.get("Z_IMAGE_TURN_WAIT_S", "900"))

STATE: dict = {"status": "loading", "error": None, "generating": False, "load_s": None, "dram": None}
MODEL: Optional[ZImageTurboTT] = None
DEVICE = None
LOCK = Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL, DEVICE
    try:
        DEVICE = open_device()
        MODEL = ZImageTurboTT(DEVICE)
        STATE.update(status="ok", load_s=round(MODEL.load_s, 2), dram=dram_stats(DEVICE))
    except Exception as exc:
        STATE.update(status="error", error=str(exc))
    try:
        yield
    finally:
        MODEL = None
        if DEVICE is not None:
            close_device(DEVICE)


app = FastAPI(title="Z-Image-Turbo on p100a", lifespan=lifespan, docs_url=None, redoc_url=None)


class Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str = Field(min_length=1, max_length=4000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    width: int = 1024
    height: int = 1024
    num_steps: Optional[int] = None
    images: Optional[list] = None


def _check(request: Request) -> None:
    if request.images:
        raise ValueError(f"{HF_REPO} is text-to-image only; reference images are not supported")
    if request.num_steps is not None and request.num_steps != SCHED.steps:
        raise ValueError(f"{HF_REPO} runs {SCHED.steps} steps")
    for axis, value in (("width", request.width), ("height", request.height)):
        if not MIN_DIM <= value <= MAX_DIM:
            raise ValueError(f"{axis} must be between {MIN_DIM} and {MAX_DIM}")
        if value % DIM_STEP:
            raise ValueError(f"{axis} must be a multiple of {DIM_STEP}")
    if request.width * request.height > MAX_PIXELS:
        raise ValueError(f"At most {MAX_PIXELS} pixels")


@app.get("/health")
def health():
    return dict(STATE)


@app.get("/info")
def info():
    return dict(STATE) | {
        "model": HF_REPO,
        "weights_revision": HF_REVISION,
        "license": LICENSE,
        "device": "Tenstorrent Blackhole p100a",
        "num_steps": SCHED.steps,
        "task_modes": ["text-to-image"],
        "resolution_limits": {"min": MIN_DIM, "max": MAX_DIM, "step": DIM_STEP, "max_pixels": MAX_PIXELS},
    }


@app.post("/predict")
def predict(request: Request):
    if not request.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    try:
        _check(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if STATE["status"] != "ok" or MODEL is None:
        raise HTTPException(503, STATE["error"] or "Model is not ready")
    if not LOCK.acquire(timeout=TURN_WAIT_S):
        raise HTTPException(409, "Another image is in progress")
    try:
        STATE["generating"] = True
        t0 = time.perf_counter()
        out, timing = MODEL.generate(request.prompt, request.width, request.height, request.seed)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "width": out.width,
            "height": out.height,
            "model": HF_REPO,
            "weights_revision": HF_REVISION,
            "license": LICENSE,
            "seed": request.seed,
            "num_steps": SCHED.steps,
            "timing_ms": {k: round(v * 1000, 2) for k, v in timing.values.items()}
                         | {"total_ms": round((time.perf_counter() - t0) * 1000, 2)},
        }
    except Exception as exc:
        STATE.update(status="error", error=str(exc))
        raise HTTPException(503, str(exc)) from exc
    finally:
        STATE["generating"] = False
        LOCK.release()
