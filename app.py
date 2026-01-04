from typing import List
import torch


from pydantic import BaseModel, Field
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, HTTPException
from typing import Optional

from model import RulLstm
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

#input_size = 24 # Sensors (21) + Settings (3)
#model = EngineRULPredictor(input_size=input_size, hidden_size=512, num_layers=2, dropout=0.2)

#model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))

#model.eval()
model_path = 'models/lstm_model.pth'

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
    input_size = 18 # Sensors (21) + Settings (3)

    model = RulLstm(n_features=input_size, hidden_size=128, num_layers=2, dropout=0.2)

    model 

    model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()

    app.state.model = model
    app.state.model.eval()
    
    app.state.scaler = joblib.load('models/scaler.pkl') # <--- 2. Load the scaler

    col_names = ['unit_nr', 'time_cycles',
                 'setting_1', 'setting_2', 'setting_3'] + [f"s_{i}" for i in range(1, 22)]
    raw_df = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
    app.state.sensor_df = raw_df


    yield  # This separates startup from shutdown
    
    # 2. Shutdown logic goes here
    print("Shutting down...")

# We pass the lifespan function to the FastAPI app
app = FastAPI(lifespan=lifespan)

if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


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

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend not built")


@app.post("/predict/")
async def predict_rul(request: Request, payload: EngineRequest):
    max_rul = 542  # use the same value you trained with

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
    col_names = index_names + setting_names + sensor_names
    df = pd.read_csv("data/test_FD001.txt", sep="\s+", header=None, names=col_names)
    df = df[df.unit_nr == payload.unit_nr]

    df = df.drop(columns=["unit_nr", "time_cycles"])  # Drop non-feature columns for scaling
    scaler = request.app.state.scaler
    scaled = scaler.transform(df)
    df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    features_to_drop = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
    df = df.drop(columns=features_to_drop)

    input_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    model = request.app.state.model

    # Perform prediction
    with torch.no_grad():
        length = int(input_tensor.shape[0])

        prediction = model(input_tensor, lengths=torch.tensor([length]))
        print(f"Raw model prediction: {prediction.item()}")
        scaled_prediction = prediction.item() * max_rul
    
    return {"predicted_rul": scaled_prediction,
            "prediction_raw": prediction.item(),
            "unit_nr": payload.unit_nr,
            "data": df.to_dict(orient="records")}


@app.post("/predict_old/")
async def predict_rul_old(request: Request):
    max_rul = 542  # use the same value you trained with

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
    col_names = index_names + setting_names + sensor_names
    df = pd.read_csv("data/train_FD001.txt", sep="\s+", header=None, names=col_names)
    df = df.head(50)  # Use only the first 50 rows for prediction

    # Convert payload to DataFrame
    #df = payload.to_dataframe()

    df = df.drop(columns=["unit_nr", "time_cycles"])  # Drop non-feature columns for scaling
    
    #features_to_drop = ['unit_nr', 'time_cycles', "s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
    #df = df.drop(columns=features_to_drop)
    scaler = request.app.state.scaler
    scaled = scaler.transform(df)
    df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    features_to_drop = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
    df = df.drop(columns=features_to_drop)

    input_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    model = request.app.state.model

    # Perform prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        #preds_real = prediction.cpu().numpy().flatten() * max_rul
        print(f"Raw model prediction: {prediction.item()}")
        scaled_prediction = prediction.item() * max_rul

    # Assuming the model outputs a single RUL value
    #rul_prediction = prediction.item()
    
    return {"predicted_rul": scaled_prediction}


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