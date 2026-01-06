import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import zipfile
import subprocess
from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset import CmapssRandomCropDataset, CmapssFullDataset
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
import os
import random
import torch
import numpy as np
import pandas as pd
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
from model import RulLstm
import matplotlib.pyplot as plt
import copy
import math


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def download_cmapss_data(data_dir: Path):
    zip_path = data_dir / "CMAPSSData.zip"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        subprocess.run(
            ["wget", "https://data.nasa.gov/docs/legacy/CMAPSSData.zip", "-O", str(zip_path)],
            check=True
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    print("Data downloaded and extracted.")

def load_fd(fd_tag: str, data_dir: Path, col_names) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / f"train_{fd_tag}.txt"
    test_path = data_dir / f"test_{fd_tag}.txt"
    rul_path = data_dir / f"RUL_{fd_tag}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    rul_df = pd.read_csv(rul_path, header=None, names=["RUL_truth"])

    max_cycle = train_df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
    train_df = train_df.merge(max_cycle, left_on="unit_nr", right_index=True)
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_cycles"]

    train_df["fd"] = fd_tag
    test_df["fd"] = fd_tag
    rul_df["fd"] = fd_tag

    return train_df, test_df, rul_df

def make_global_unit_ids(train_df: pd.DataFrame, test_df: pd.DataFrame, next_unit: int):
    uniq_units_train = sorted(train_df["unit_nr"].unique())
    mapping_train = {u: next_unit + i for i, u in enumerate(uniq_units_train)}

    train_df = train_df.assign(
        unit_nr_orig=train_df["unit_nr"],
        unit_nr=train_df["unit_nr"].map(mapping_train),
    )

    test_df = test_df.assign(
        unit_nr_orig=test_df["unit_nr"],
        unit_nr=test_df["unit_nr"] + next_unit - 1,
    )

    next_unit = next_unit + len(uniq_units_train)
    return train_df, test_df, next_unit

def split_by_unit(df: pd.DataFrame, val_frac: float, seed: int):
    units = df["unit_nr"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)

    n_val = int(np.ceil(len(units) * val_frac))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])

    train_split = df[df["unit_nr"].isin(train_units)].copy()
    val_split = df[df["unit_nr"].isin(val_units)].copy()

    return train_split, val_split, sorted(train_units), sorted(val_units)



def concatenate_fd_data(fd_tags: list[str], data_dir: Path, col_names):
    train_parts, test_parts, rul_parts = [], [], []
    next_unit = 1

    for fd_tag in fd_tags:
        tr, te, rul = load_fd(fd_tag, data_dir, col_names)
        tr, te, next_unit = make_global_unit_ids(tr, te, next_unit)
        train_parts.append(tr)
        test_parts.append(te)
        rul_parts.append(rul)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    rul_labels_df = pd.concat(rul_parts, ignore_index=True)
    return train_df, test_df, rul_labels_df

def scale_data(train_split_df: pd.DataFrame, val_split_df: pd.DataFrame, test_df: pd.DataFrame):

    not_scaled_cols = ['unit_nr', 'RUL', 'max_cycle', 'time_cycles']

    col_set = train_split_df.columns
    columns_to_scale = [col for col in col_set if col not in not_scaled_cols]

    print("Columns to scale:", columns_to_scale)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_split = scaler.fit_transform(train_split_df[columns_to_scale])
    scaled_train_split_df = pd.DataFrame(scaled_train_split, columns=columns_to_scale, index=train_split_df.index)
    scaled_val_split = scaler.transform(val_split_df[columns_to_scale])
    scaled_val_split_df = pd.DataFrame(scaled_val_split, columns=columns_to_scale, index=val_split_df.index)
    scaled_test = scaler.transform(test_df[columns_to_scale])
    scaled_test_df = pd.DataFrame(scaled_test, columns=columns_to_scale, index=test_df.index)


    scaled_train_split_df.insert(0, 'unit_nr', train_split_df['unit_nr'])
    scaled_train_split_df.insert(1, 'time_cycles', train_split_df['time_cycles'])
    scaled_train_split_df.insert(len(scaled_train_split_df.columns), 'RUL', train_split_df['RUL'])
    scaled_train_split_df.insert(len(scaled_train_split_df.columns), 'max_cycle', train_split_df['max_cycle'])

    scaled_val_split_df.insert(0, 'unit_nr', val_split_df['unit_nr'])
    scaled_val_split_df.insert(1, 'time_cycles', val_split_df['time_cycles'])
    scaled_val_split_df.insert(len(scaled_val_split_df.columns), 'RUL', val_split_df['RUL'])
    scaled_val_split_df.insert(len(scaled_val_split_df.columns), 'max_cycle', val_split_df['max_cycle'])

    scaled_test_df.insert(0, 'unit_nr', test_df['unit_nr'])
    scaled_test_df.insert(1, 'time_cycles', test_df['time_cycles'])

    return scaled_train_split_df, scaled_val_split_df, scaled_test_df

def convert_to_datasets(scaled_train_split_df: pd.DataFrame, scaled_val_split_df: pd.DataFrame, scaled_test_df: pd.DataFrame, rul_labels_df: pd.DataFrame):
    sequences_by_unit = {}
    rul_by_unit = {}

    non_train_feature_cols = ['unit_nr', 'time_cycles', 'RUL', 'max_cycle']
    feature_cols = [c for c in scaled_train_split_df.columns if c not in non_train_feature_cols]

    print("Feature columns:", feature_cols,"count features:" ,len(feature_cols))

    def build_unit_dicts(df, feature_cols=feature_cols):
        sequences_by_unit = {}
        rul_by_unit = {}

        df = df.sort_values(["unit_nr", "time_cycles"])

        for unit_id, g in df.groupby("unit_nr", sort=False):
            x = torch.tensor(g[feature_cols].to_numpy(), dtype=torch.float32)
            y = torch.tensor(g["RUL"].to_numpy(), dtype=torch.float32)

            sequences_by_unit[int(unit_id)] = x
            rul_by_unit[int(unit_id)] = y

        return sequences_by_unit, rul_by_unit

    train_sequences_by_unit, train_rul_by_unit = build_unit_dicts(scaled_train_split_df)
    val_sequences_by_unit, val_rul_by_unit = build_unit_dicts(scaled_val_split_df)

    train_unit_nr = list(train_sequences_by_unit.keys())[0]
    val_unit_nr = list(val_sequences_by_unit.keys())[0]
    print("train_sequences_by_unit", len(train_sequences_by_unit), "unit shape", train_sequences_by_unit[train_unit_nr].shape)
    print("train_rul_by_unit", len(train_rul_by_unit), "unit shape", train_rul_by_unit[train_unit_nr].shape)
    print("val_sequences_by_unit", len(val_sequences_by_unit), "unit shape", val_sequences_by_unit[val_unit_nr].shape)
    print("val_rul_by_unit", len(val_rul_by_unit), "unit shape", val_rul_by_unit[val_unit_nr].shape)

    non_train_feature_cols = ['unit_nr', 'time_cycles', 'RUL', 'max_cycle']
    feature_cols = [c for c in scaled_train_split_df.columns if c not in non_train_feature_cols]

    print("Feature columns:", feature_cols,"count features:" ,len(feature_cols))

    def build_test_sequences(df):
        sequences = {}
        df = df.sort_values(["unit_nr", "time_cycles"])

        for unit_id, g in df.groupby("unit_nr", sort=False):
            x = torch.tensor(g[feature_cols].to_numpy(), dtype=torch.float32)
            sequences[int(unit_id)] = x

        return sequences

    test_sequences_by_unit = build_test_sequences(scaled_test_df)

    rul_truth = rul_labels_df["RUL_truth"].to_numpy().astype(float)
    test_unit_ids = sorted(test_sequences_by_unit.keys())

    test_targets_by_unit = {unit_id: float(rul_truth[i]) for i, unit_id in enumerate(test_unit_ids)}
    print("test_sequences_by_unit", len(test_sequences_by_unit))
    print("test_sequences_by_unit[0]", test_sequences_by_unit[1].shape)

    return (
        train_sequences_by_unit,
        train_rul_by_unit,
        val_sequences_by_unit,
        val_rul_by_unit,
        test_sequences_by_unit,
        test_targets_by_unit,
        feature_cols,
    )

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    #features_to_drop = ['s_1', 's_18', 'unit_nr_orig', 's_10', 'setting_3', 's_19', 's_5', 's_14', 's_16', 'fd']

    features_to_drop = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19", "unit_nr_orig", "setting_3", "fd"]
    df = df.drop(columns=features_to_drop)
    return df



# Pad sequence windows as they can have variable lengths
def collate_pad(batch: List[Tuple[torch.Tensor, float, int]]):
    seqs, targets, lengths = zip(*batch)

    lengths_t = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    return padded.float(), lengths_t, targets_t


def make_loader(sequences_by_unit, targets_by_unit, samples_per_epoch, batch_size, l_min, l_max, num_workers=0, train=True):
    drop_last = True
    
    if train:
        ds = CmapssRandomCropDataset(
            sequences_by_unit=sequences_by_unit,
            targets_by_unit=targets_by_unit,
            samples_per_epoch=samples_per_epoch,
            l_min=l_min,
            l_max=l_max,
        )
    else:
        drop_last = False

        ds = CmapssFullDataset(
        sequences_by_unit=sequences_by_unit,
        targets_by_unit=targets_by_unit,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pad,
        drop_last=drop_last,
        pin_memory=True,
    )

    return loader





def run_epoch(model, loader, loss_fn, device, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    if train:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()

    with context:
        for padded, lengths, targets in loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            preds = model(padded, lengths)
            loss = loss_fn(preds, targets)

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

    return total_loss / max(total_samples, 1)



def train_full(epochs, model, train_loader, val_loader, optimizer, device):
    if not os.path.exists('models'):
        os.makedirs('models')
    
    loss_fn = nn.SmoothL1Loss(reduction="mean")

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            train=True,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            train=False,
            optimizer=None,
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, 'models/lstm_model.pth')
        print("Model weights saved to models/lstm_model.pth")

    return best_val


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_trial(params,  train_sequences_by_unit, train_rul_by_unit, val_sequences_by_unit, val_rul_by_unit, feature_cols, EPOCHS=20, device=torch.device("cpu"), seed=0):
    set_seed(seed)

    train_loader = make_loader(
        train_sequences_by_unit,
        train_rul_by_unit,
        samples_per_epoch=params["samples_per_epoch"],
        batch_size=params["batch_size"],
        l_min=params["l_min"],
        l_max=params["l_max"],
    )
    val_loader = make_loader(
        val_sequences_by_unit,
        val_rul_by_unit,
        samples_per_epoch=params.get("val_samples_per_epoch", 14000),
        batch_size=params["batch_size"],
        l_min=params["l_min"],
        l_max=params.get("val_l_max", params["l_max"]),
    )

    model = RulLstm(
        n_features=len(feature_cols),
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    return train_full(EPOCHS, model, train_loader, val_loader, optimizer, device)

def main():
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    FD_TAGS = ["FD001"]
    EPOCHS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    download_cmapss_data(data_dir=DATA_DIR)

    index_names = ["unit_nr", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = [f"s_{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    print("Column names:", col_names)


    train_df, test_df, rul_labels_df = concatenate_fd_data(FD_TAGS, DATA_DIR, col_names)
    train_df = drop_unused_columns(train_df)
    test_df = drop_unused_columns(test_df)
    
    train_split_df, val_split_df, train_units, val_units = split_by_unit(train_df, val_frac=0.2, seed=42)
    print(f"Train units: {len(train_units)}, Val units: {len(val_units)}")
    print("train_split_df.columns:", train_split_df.columns.tolist())
    
    scaled_train_split_df, scaled_val_split_df, scaled_test_df = scale_data(train_split_df, val_split_df, test_df)
    (
        train_sequences_by_unit,
        train_rul_by_unit,
        val_sequences_by_unit,
        val_rul_by_unit,
        test_sequences_by_unit,
        test_targets_by_unit,
        feature_cols,
    ) = convert_to_datasets(scaled_train_split_df, scaled_val_split_df, scaled_test_df, rul_labels_df)
    print("Datasets prepared.")


    search_space = {
        "hidden_size": [64],
        "num_layers": [1, 2],
        "dropout": [0.0,0.3],
        "lr": [1e-3, 2e-3, 5e-4],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [16, 32],
        "samples_per_epoch": [60000],
        "l_min": [30],
        "l_max": [200],
    }


    keys = list(search_space.keys())
    candidates = [dict(zip(keys, vals)) for vals in itertools.product(*(search_space[k] for k in keys))]

    EPOCHS = 2  # fast sweep; raise later for final training
    best = (float("inf"), None)
    results = []

    results_path = Path("tuning_results.txt")
    with results_path.open("w", encoding="utf-8") as f:
        f.write("trial\tval_loss\tparams\n")

        for i, params in enumerate(candidates):
            print(f"Trial {i+1}/{len(candidates)} with params: {params}")
            val_loss = run_trial(
                params,
                train_sequences_by_unit,
                train_rul_by_unit,
                val_sequences_by_unit,
                val_rul_by_unit,
                feature_cols,
                EPOCHS=EPOCHS,
                device=device,
                seed=123 + i,
            )
            results.append({**params, "val_loss": val_loss})
            if val_loss < best[0]:
                best = (val_loss, params)

            f.write(f"{i+1}\t{val_loss:.6f}\t{params}\n")
            f.flush()

    print("best:", best)






if __name__ == "__main__":
    main()
