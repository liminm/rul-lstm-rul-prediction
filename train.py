
import argparse
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import CmapssRandomCropDataset, CmapssLastWindowDataset, CmapssFullDataset
from model import RulLstm

# --- Constants & Config ---
SEED = 42
FD_TAGS = ["FD001"]
COL_NAMES = ["unit_nr", "time_cycles"] + [f"setting_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]
FEATURES_TO_DROP = ['s_1', 's_18', 'unit_nr_orig', 's_10', 'setting_3', 's_19', 's_5', 's_14', 's_16', 'fd']

# --- Helper Functions ---

def load_fd(fd_tag: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, test, and RUL files for a given FD tag."""
    train_path = data_dir / f"train_{fd_tag}.txt"
    test_path = data_dir / f"test_{fd_tag}.txt"
    rul_path = data_dir / f"RUL_{fd_tag}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COL_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COL_NAMES)
    rul_df = pd.read_csv(rul_path, header=None, names=["RUL_truth"])

    # Calculate RUL for train
    max_cycle = train_df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
    train_df = train_df.merge(max_cycle, left_on="unit_nr", right_index=True)
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_cycles"]

    train_df["fd"] = fd_tag
    test_df["fd"] = fd_tag
    rul_df["fd"] = fd_tag

    return train_df, test_df, rul_df

def make_global_unit_ids(train_df: pd.DataFrame, test_df: pd.DataFrame, fd_tag: str, next_unit: int):
    """Re-maps unit IDs to be global across FD tags."""
    uniq_units_train = sorted(train_df["unit_nr"].unique())
    mapping_train = {u: next_unit + i for i, u in enumerate(uniq_units_train)}

    train_df = train_df.assign(
        unit_nr_orig=train_df["unit_nr"],
        unit_nr=train_df["unit_nr"].map(mapping_train),
    )

    test_df = test_df.assign(
        unit_nr_orig=test_df["unit_nr"],
        unit_nr=test_df["unit_nr"] + next_unit - 1, # TODO: verify this logic from notebook carefully
    )
    
    # Correction based on notebook logic:
    # test units follow train units logic but here we just need unique IDs.
    # The notebook logic was:
    # test_df = test_df.assign(
    #    unit_nr_orig=test_df["unit_nr"],
    #    unit_nr=test_df["unit_nr"] + next_unit - 1, 
    # )
    # This seems to assume test unit IDs restart at 1? Yes usually.
    # But wait, next_unit is updated after processing train.
    
    next_unit = next_unit + len(uniq_units_train)
    return train_df, test_df, next_unit

def split_by_unit(df: pd.DataFrame, val_frac: float, seed: int):
    """Splits dataframe into train and validation sets by unit ID."""
    units = df["unit_nr"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)

    n_val = int(np.ceil(len(units) * val_frac))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])

    train_split = df[df["unit_nr"].isin(train_units)].copy()
    val_split = df[df["unit_nr"].isin(val_units)].copy()

    return train_split, val_split, sorted(list(train_units)), sorted(list(val_units))

def build_unit_dicts(df, feature_cols):
    """Groups dataframe by unit_nr and creates tensors for features and targets."""
    sequences_by_unit = {}
    rul_by_unit = {}

    df = df.sort_values(["unit_nr", "time_cycles"])

    for unit_id, g in df.groupby("unit_nr", sort=False):
        x = torch.tensor(g[feature_cols].to_numpy(), dtype=torch.float32)
        y = torch.tensor(g["RUL"].to_numpy(), dtype=torch.float32)

        sequences_by_unit[int(unit_id)] = x
        rul_by_unit[int(unit_id)] = y

    return sequences_by_unit, rul_by_unit

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    """
    Collate function to pad variable length sequences.
    batch is a list of tuples (seq, target, length)
    """
    sequences, targets, lengths = zip(*batch)
    
    # Pad sequences: returns (batch, max_len, features) if batch_first=True
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_seqs, targets, lengths

# --- Main Training Script ---

def main():
    parser = argparse.ArgumentParser(description="Train RUL LSTM")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to CMAPSS data")
    parser.add_argument("--data-tags", nargs="+", default=["FD001"], help="List of FD tags to train on (e.g. FD001 FD002)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--l-min", type=int, default=20)
    parser.add_argument("--l-max", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    
    # 1. Load Data
    print(f"Loading data for tags: {args.data_tags}...")
    train_dfs = []
    test_dfs = []
    next_unit = 0

    for fd_tag in args.data_tags:
        train_df, test_df, rul_df = load_fd(fd_tag, data_dir)
        train_df, test_df, next_unit = make_global_unit_ids(train_df, test_df, fd_tag, next_unit)
        train_dfs.append(train_df)
        test_dfs.append(train_df) # Wait, is this right? No, test_dfs.append(test_df)
    
    # Based on notebook logic, we concat all
    train_df_all = pd.concat(train_dfs, ignore_index=True)
    # test_df_all = pd.concat(test_dfs, ignore_index=True) # Not strictly needed for training script unless calculating test score

    if args.debug:
        print("[DEBUG] Reducing dataset size")
        # Keep only first 10 units
        units = train_df_all["unit_nr"].unique()[:10]
        train_df_all = train_df_all[train_df_all["unit_nr"].isin(units)].copy()

    # 2. Feature Selection
    print("Preprocessing...")
    valid_drop_cols = [c for c in FEATURES_TO_DROP if c in train_df_all.columns]
    train_df_all = train_df_all.drop(columns=valid_drop_cols)
    
    # 3. Train/Val Split
    print("Splitting train/val...")
    train_split_df, val_split_df, train_units, val_units = split_by_unit(train_df_all, val_frac=args.val_frac, seed=SEED)
    print(f"Train units: {len(train_units)}, Val units: {len(val_units)}")

    # 4. Scaling
    print("Scaling...")
    not_scaled_cols = ['unit_nr', 'RUL', 'max_cycle', 'time_cycles']
    col_set = train_split_df.columns
    columns_to_scale = [col for col in col_set if col not in not_scaled_cols]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_vals = scaler.fit_transform(train_split_df[columns_to_scale])
    scaled_val_vals = scaler.transform(val_split_df[columns_to_scale])

    scaled_train_df = pd.DataFrame(scaled_train_vals, columns=columns_to_scale, index=train_split_df.index)
    scaled_val_df = pd.DataFrame(scaled_val_vals, columns=columns_to_scale, index=val_split_df.index)

    # Re-insert non-scaled columns
    for df_scaled, df_orig in [(scaled_train_df, train_split_df), (scaled_val_df, val_split_df)]:
        df_scaled['unit_nr'] = df_orig['unit_nr']
        df_scaled['time_cycles'] = df_orig['time_cycles']
        df_scaled['RUL'] = df_orig['RUL']
        df_scaled['max_cycle'] = df_orig['max_cycle']

    feature_cols = columns_to_scale
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # 5. Build Datasets
    print("Building datasets...")
    train_seqs, train_ruls = build_unit_dicts(scaled_train_df, feature_cols)
    val_seqs, val_ruls = build_unit_dicts(scaled_val_df, feature_cols)

    train_dataset = CmapssRandomCropDataset(
        sequences_by_unit=train_seqs,
        targets_by_unit=train_ruls,
        samples_per_epoch=args.samples_per_epoch if not args.debug else 100,
        l_min=args.l_min,
        l_max=args.l_max
    )
    
    val_dataset = CmapssRandomCropDataset(
        sequences_by_unit=val_seqs,
        targets_by_unit=val_ruls,
        samples_per_epoch=2000 if not args.debug else 50,
        l_min=args.l_min,
        l_max=args.l_max
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 6. Model Setup
    model = RulLstm(
        n_features=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    
    # 7. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        steps = 0
        
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device) # lengths needed for packing?
            # Note: RulLstm forward takes (padded, lengths)
            # lengths needs to be on CPU for pack_padded_sequence in older pytorch, 
            # but model.py has `lengths_sorted.cpu()` so passing gpu tensor is fine?
            # Use model.py logic: `lengths_sorted.cpu()` is called inside.
            
            optimizer.zero_grad()
            preds = model(X, lengths)
            loss = criterion(preds, y.float()) # y is target
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
        
        avg_train_loss = train_loss / steps if steps > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for X, y, lengths in val_loader:
                X, y, lengths = X.to(device), y.to(device), lengths.to(device)
                preds = model(X, lengths)
                loss = criterion(preds, y.float())
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  New best model saved.")

if __name__ == "__main__":
    main()
