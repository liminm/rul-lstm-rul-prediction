import onnxruntime as ort
import numpy as np
import pandas as pd


def run_onnx_rul_inference(request, payload, model_path):

    max_rul = 542  # use the same value you trained with

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
    col_names = index_names + setting_names + sensor_names
    
    df = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
    df = df[df.unit_nr == payload.unit_nr]
    observed_cycles = None
    if not df.empty:
        observed_cycles = int(df["time_cycles"].max())

    rul_df = pd.read_csv("data/RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"])
    rul_df.index = range(1, len(rul_df) + 1)
    rul_df.index.name = "unit_nr"
    rul_df = rul_df[rul_df.index == payload.unit_nr]
    true_rul = None
    if not rul_df.empty:
        true_rul = float(rul_df["rul"].iloc[0])

    features_to_drop = ["unit_nr", "time_cycles", "setting_3", "s_1", "s_5", "s_10", "s_14", "s_16", "s_18", "s_19"]

    df = df.drop(columns=features_to_drop)  # Drop non-feature columns for scaling
    scaler = request.app.state.scaler
    scaled = scaler.transform(df)
    df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    input_np = df.to_numpy().astype(np.float32)
    input_np = np.expand_dims(input_np, axis=0)  # Add batch dimension  

    # Load ONNX model
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Perform prediction
    inputs = {
        "input": input_np,
        "lengths": np.array([input_np.shape[1]], dtype=np.int64)
    }
    outputs = ort_session.run(None, inputs)

    prediction = outputs[0].item()

    print(f"Raw ONNX model prediction: {prediction}")

    rul_dist = getattr(request.app.state, "rul_dist", None)
    rul_stats = getattr(request.app.state, "rul_stats", None)
    train_cycle_stats = getattr(request.app.state, "train_cycle_stats", None)

    rul_percentile = None
    if isinstance(rul_dist, np.ndarray) and rul_dist.size > 0:
        rul_percentile = float(
            np.searchsorted(rul_dist, prediction, side="right")
            / rul_dist.size
            * 100
        )

    health_band = None
    health_band_level = None
    if rul_stats and rul_stats.get("p20") is not None:
        if prediction <= rul_stats["p20"]:
            health_band = "Critical"
            health_band_level = "critical"
        elif prediction <= rul_stats["p50"]:
            health_band = "Monitor"
            health_band_level = "monitor"
        elif prediction <= rul_stats["p80"]:
            health_band = "Healthy"
            health_band_level = "healthy"
        else:
            health_band = "Excellent"
            health_band_level = "excellent"

    context = {
        "rul_percentile": rul_percentile,
        "health_band": health_band,
        "health_band_level": health_band_level,
        "observed_cycles": observed_cycles,
        "train_rul_min": rul_stats.get("min") if rul_stats else None,
        "train_rul_max": rul_stats.get("max") if rul_stats else None,
        "train_rul_median": rul_stats.get("p50") if rul_stats else None,
        "train_cycle_median": train_cycle_stats.get("median") if train_cycle_stats else None,
        "train_cycle_mean": train_cycle_stats.get("mean") if train_cycle_stats else None,
    }

    return prediction, true_rul, context
