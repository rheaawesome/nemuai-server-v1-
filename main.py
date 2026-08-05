# main.py -- FastAPI server for the NemuAI inference pipeline.
#
# Loads the model bundle once at startup (loading YAMNet per-request would be
# very slow), then exposes:
#   GET  /health            -> quick check that the bundle loaded correctly
#   POST /analyze           -> upload an audio file, get back the same report
#                               JSON schema the original notebook produces
#
# Run locally:
#   1. Put nemuai_final_model.joblib, nemuai_final_scaler.joblib,
#      nemuai_final_threshold.json, nemuai_final_config.json,
#      nemuai_final_selection.json, and README.txt into ./model_bundle/
#   2. pip install -r requirements.txt  (also needs system ffmpeg -- see README.md)
#   3. uvicorn main:app --reload --port 8000
#   4. curl -F "audio=@america.m4a" http://localhost:8000/analyze

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import nemuai_pipeline as pipeline

MODEL_DIR = Path(os.environ.get("NEMUAI_MODEL_DIR", Path(__file__).parent / "model_bundle"))

bundle_state = {"bundle": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle_state["bundle"] = pipeline.load_bundle(MODEL_DIR)
        print(f"NemuAI bundle loaded from {MODEL_DIR} "
              f"({bundle_state['bundle'].config['model_name']}, "
              f"threshold={bundle_state['bundle'].event_threshold})")
    except Exception as exc:  # noqa: BLE001 -- surface any load failure via /health instead of crashing import
        bundle_state["error"] = str(exc)
        print(f"WARNING: NemuAI bundle failed to load from {MODEL_DIR}: {exc}")
    yield


app = FastAPI(title="NemuAI Server", lifespan=lifespan)


@app.get("/health")
def health():
    if bundle_state["bundle"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_dir": str(MODEL_DIR), "error": bundle_state["error"]},
        )
    return {
        "status": "ok",
        "model_name": bundle_state["bundle"].config["model_name"],
        "model_version": bundle_state["bundle"].config["model_version"],
        "event_threshold": bundle_state["bundle"].event_threshold,
    }


@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    if bundle_state["bundle"] is None:
        raise HTTPException(status_code=503, detail=f"Model bundle not loaded: {bundle_state['error']}")

    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in pipeline.SUPPORTED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Supported: {sorted(pipeline.SUPPORTED_AUDIO_SUFFIXES)}",
        )

    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / (audio.filename or f"upload{suffix}")
        tmp_path.write_bytes(contents)
        try:
            report = pipeline.run_inference(tmp_path, bundle_state["bundle"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return report
