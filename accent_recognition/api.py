"""FastAPI app; audio is processed in memory and is not persisted."""

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import MAX_BYTES, AudioError
from .model import AccentModel

STATIC = Path(__file__).resolve().parent.parent / "static"
MODEL_PATH = Path(os.getenv("ACCENT_MODEL_PATH", "models/accent.joblib"))


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    app = FastAPI(title="Accent Recognition", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.state.model = None
    app.state.model_error = None
    if model_path.is_file():
        try:
            app.state.model = AccentModel.load(model_path)
        except (ValueError, OSError, ImportError) as exc:
            app.state.model_error = str(exc)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ready" if app.state.model else "model_unavailable",
                "model_loaded": app.state.model is not None,
                "labels": app.state.model.labels if app.state.model else []}

    @app.post("/api/predict")
    async def predict(audio: UploadFile = File(...)) -> dict:
        if app.state.model is None:
            raise HTTPException(503, "Model unavailable. Train a model and restart the server.")
        if audio.content_type not in {"audio/wav", "audio/x-wav", "audio/wave", "audio/flac",
                                      "audio/ogg", "audio/mpeg", "application/octet-stream"}:
            raise HTTPException(415, "Upload a WAV, FLAC, OGG, or MP3 audio file.")
        content = await audio.read(MAX_BYTES + 1)
        await audio.close()
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "Audio file is larger than 12 MB.")
        try:
            return app.state.model.predict(content)
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
