from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse

from predict import run_onnx_rul_inference

MODEL_PATH = Path("models/lstm_model.onnx")
DATA_DIR = Path("data")
STATIC_DIR = Path(__file__).parent / "static"
COL_NAMES = (
    ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)


class EngineData(BaseModel):
    unit_nr: int
    time_cycles: int
    setting_1: float
    setting_2: float
    setting_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float


class EngineRequest(BaseModel):
    unit_nr: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")

    app.state.scaler = joblib.load("models/scaler.pkl")

    raw_df = pd.read_csv(
        DATA_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COL_NAMES
    )
    app.state.sensor_df = raw_df
    engine_ids = sorted(raw_df["unit_nr"].unique().tolist())
    app.state.engine_ids = engine_ids
    app.state.engine_id_set = set(engine_ids)

    train_df = pd.read_csv(
        DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COL_NAMES
    )
    max_cycle = train_df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
    train_df = train_df.merge(max_cycle, left_on="unit_nr", right_index=True)
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_cycles"]
    rul_values = train_df["RUL"].to_numpy(dtype=float)

    if rul_values.size > 0:
        rul_sorted = np.sort(rul_values)
        app.state.rul_dist = rul_sorted
        app.state.rul_stats = {
            "min": float(np.min(rul_values)),
            "max": float(np.max(rul_values)),
            "p20": float(np.percentile(rul_values, 20)),
            "p50": float(np.percentile(rul_values, 50)),
            "p80": float(np.percentile(rul_values, 80)),
        }
    else:
        app.state.rul_dist = np.array([])
        app.state.rul_stats = {
            "min": None,
            "max": None,
            "p20": None,
            "p50": None,
            "p80": None,
        }

    app.state.train_cycle_stats = {
        "median": float(max_cycle.median()),
        "mean": float(max_cycle.mean()),
        "max": float(max_cycle.max()),
    }

    yield  # This separates startup from shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173",  # The address of your React App
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],
)


@app.post("/predict/", response_model=dict)
async def predict(request: Request, payload: EngineRequest):
    engine_id_set = getattr(request.app.state, "engine_id_set", None)
    if engine_id_set is None:
        engine_id_set = set(request.app.state.sensor_df["unit_nr"].unique().tolist())
    if payload.unit_nr not in engine_id_set:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {payload.unit_nr} not found",
        )
    prediction, true_rul, context = run_onnx_rul_inference(
        request, payload, MODEL_PATH.as_posix()
    )
    return {
        "predicted_rul": int(prediction),
        "true_rul": None if true_rul is None else int(true_rul),
        "unit_nr": payload.unit_nr,
        "context": context,
    }


@app.get("/sensors/", response_model=List[EngineData])
async def list_sensors(
    request: Request,
    limit: int = Query(50, gt=0, le=1000),
    unit_nr: int = Query(1, ge=1),
    start_cycle: Optional[int] = Query(None, ge=0),
    end_cycle: Optional[int] = Query(None, ge=0),
):
    df = request.app.state.sensor_df
    df = df[df["unit_nr"] == unit_nr]
    if start_cycle is not None and end_cycle is not None and end_cycle < start_cycle:
        raise HTTPException(status_code=400, detail="end_cycle must be >= start_cycle")
    if start_cycle is not None:
        df = df[df["time_cycles"] >= start_cycle]
    if end_cycle is not None:
        df = df[df["time_cycles"] <= end_cycle]
    return df.head(limit).to_dict(orient="records")


@app.get("/engines/", response_model=List[int])
async def list_engines(request: Request):
    engine_ids = getattr(request.app.state, "engine_ids", None)
    if engine_ids is None:
        df = request.app.state.sensor_df
        engine_ids = df["unit_nr"].unique().tolist()
    return engine_ids

if STATIC_DIR.exists():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/vite.svg")
async def vite_svg():
    svg = STATIC_DIR / "vite.svg"
    if svg.exists():
        return FileResponse(svg)
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if full_path.startswith(("engines", "sensors", "predict", "docs", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not found")

    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Frontend not built")
