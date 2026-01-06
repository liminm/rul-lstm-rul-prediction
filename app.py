from typing import List


from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, HTTPException
from typing import Optional

import joblib
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from predict import run_onnx_rul_inference

model_path = "models/lstm_model.onnx"

static_dir = Path(__file__).parent / "static"


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


class InferencePayload(BaseModel):
    engine_data_sequence: List[EngineData] = Field(min_length=1, max_length=50)

    def to_dataframe(self) -> pd.DataFrame:
        data_dicts = [edata.model_dump() for edata in self.engine_data_sequence]
        return pd.DataFrame(data_dicts)

class EngineRequest(BaseModel):
    unit_nr: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup logic goes here
    print("Loading model...")
    input_size = 16 # Sensors (14) + Settings (2)

    #model = RulLstm(n_features=input_size, hidden_size=128, num_layers=2, dropout=0.2)

    #model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
    #model.eval()

    #app.state.model = model
    #app.state.model.eval()
    
    app.state.scaler = joblib.load('models/scaler.pkl') # <--- 2. Load the scaler

    col_names = ['unit_nr', 'time_cycles',
                 'setting_1', 'setting_2', 'setting_3'] + [f"s_{i}" for i in range(1, 22)]
    raw_df = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
    app.state.sensor_df = raw_df

    train_df = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)
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
    
    # 2. Shutdown logic goes here
    print("Shutting down...")

# We pass the lifespan function to the FastAPI app
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
    prediction, true_rul, context = run_onnx_rul_inference(
        request, payload, model_path
    )
    return {
        "predicted_rul": int(prediction),
        "true_rul": int(true_rul),
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
    df = request.app.state.sensor_df
    engine_ids = df["unit_nr"].unique().tolist()
    return engine_ids

static_dir = Path(__file__).parent / "static"

if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/vite.svg")
async def vite_svg():
    svg = static_dir / "vite.svg"
    if svg.exists():
        return FileResponse(svg)
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if full_path.startswith(("engines", "sensors", "predict", "docs", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not found")

    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Frontend not built")
