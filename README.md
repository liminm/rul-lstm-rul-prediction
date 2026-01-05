# CMAPSS Jet Engine RUL Prediction

Predict remaining useful life (RUL) of turbofan engines from multivariate sensor telemetry. The goal is to estimate how many cycles remain before failure so maintenance can be scheduled proactively.

## Dataset
- NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation data.
- Each row is one engine cycle with 3 operational settings and 21 sensor measurements.
- Training labels are computed as `max_cycle - time_cycles`; test labels come from `RUL_FD00*.txt`.
- Data is included in `data/` (train/test/RUL text files and `CMAPSSData.zip`).
- The notebook trains on FD001 + FD003 by default; the API currently uses FD001 test data.

## Approach
- Clean and merge CMAPSS subsets, compute RUL labels, and split by engine id.
- Scale numeric features with MinMaxScaler; store the scaler in `models/scaler.pkl`.
- Drop non-informative sensors: `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`.
- Build variable-length sequences per engine and train an LSTM (`model.RulLstm`) with SmoothL1 loss.
- Hyperparameter sweeps are recorded in `tuning_results.txt`.
- Export the best model to `models/lstm_model.pth` and `models/lstm_model.onnx`.

## Repository layout
- `notebook.ipynb` - data prep, EDA, feature analysis, training, tuning, export.
- `eda.ipynb` - extra exploration.
- `model.py` - PyTorch LSTM model.
- `dataset.py` - random-crop dataset for variable-length sequences.
- `app.py` - FastAPI inference with ONNX Runtime.
- `app_torch.py` - FastAPI inference with PyTorch.
- `models/` - model weights, ONNX export, scaler.
- `sensor-dashboard/` - React dashboard for engine and sensor plots.
- `Dockerfile` - container that builds UI and serves API.

## Setup
### Python environment
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you only want to run the API, install the minimal dependencies:
```
pip install -r requirements.infer.txt
```

### Notebook (EDA + training)
Run the notebook end-to-end to reproduce preprocessing, EDA, training, tuning, and model export:
```
jupyter notebook notebook.ipynb
```

Outputs:
- `models/lstm_model.pth`
- `models/lstm_model.onnx`
- `models/scaler.pkl`

### API (FastAPI)
```
uvicorn app:app --reload --port 8080
```

Example request:
```
curl -X POST "http://localhost:8080/predict/" \
  -H "Content-Type: application/json" \
  -d '{"unit_nr": 1}'
```

Useful endpoints:
- `GET /engines` -> list engine ids
- `GET /sensors?unit_nr=1&limit=50&start_cycle=0&end_cycle=200`
- `POST /predict/` -> RUL prediction for one engine id

### Frontend dashboard
```
cd sensor-dashboard
npm install
npm run dev
```
The Vite dev server runs at `http://localhost:5173` and expects the API at `http://localhost:8080`.

### Docker
```
docker build -t cmapss-rul .
docker run --rm -p 8080:8080 cmapss-rul
```
This builds the React UI and serves it alongside the API on `http://localhost:8080`.


## Cloud deployment (Google Cloud Run)
Set your project and region:
```
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region YOUR_REGION
```

Enable required services (one-time):
```
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Create an Artifact Registry repo (one-time):
```
gcloud artifacts repositories create rul-repo   --repository-format=docker   --location=YOUR_REGION   --description="RUL app images"
```

Build and push (Cloud Build):
```
gcloud builds submit --tag YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/rul-repo/rul-app:latest
```

Deploy to Cloud Run:
```
gcloud run deploy rul-app   --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/rul-repo/rul-app:latest   --allow-unauthenticated   --port 8080
```

Fetch the service URL:
```
gcloud run services describe rul-app --format='value(status.url)'
```

Local build alternative (if you cannot use Cloud Build):
```
gcloud auth configure-docker YOUR_REGION-docker.pkg.dev

docker build -t YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/rul-repo/rul-app:latest .
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/rul-repo/rul-app:latest
```

## Notes
- The API reads `data/test_FD001.txt`. To serve other subsets, update the file path in `app.py` and retrain/export the model if needed.
- The API rescales the normalized output by `max_rul = 542` (see `app.py`).

## References
- A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008.
