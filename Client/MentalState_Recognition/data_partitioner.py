"""
Data Partitioner for EEG Mental State Recognition
Handles non-IID heterogeneous distribution of data across federated learning clients
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from collections import Counter

# Configuration
FS = 256
WIN_S = 1.0
WIN = int(FS * WIN_S)  # 256
STRIDE = 128  # 0.5 s
SEED = 42

# Band-power columns
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
CHANS = ['TP9', 'AF7', 'AF8', 'TP10']
FEAT_COLS = [f'{b}_{c}' for b in BANDS for c in CHANS]  # 20

# Labels
CLASS_ORDER = ['alerted', 'concentrated', 'drowsy', 'neutral', 'relaxed']
LBL2ID = {c: i for i, c in enumerate(CLASS_ORDER)}
ID2LBL = {i: c for c, i in LBL2ID.items()}
NUM_CLASSES = len(CLASS_ORDER)

# Set seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)


def infer_label_from_name(path: str):
    """Infer label from filename"""
    n = os.path.basename(path).lower()
    if n.startswith("alerted"): return "alerted"
    if n.startswith("concentrated"): return "concentrated"
    if n.startswith("drowsy"): return "drowsy"
    if n.startswith("neutral"): return "neutral"
    if n.startswith("relaxed"): return "relaxed"
    return None


def read_csv_safe(path: str) -> pd.DataFrame:
    """Safely read CSV file with EEG data"""
    try:
        df = pd.read_csv(path, low_memory=False, engine="c")
    except Exception:
        df = pd.read_csv(path, engine="python")
    
    miss = [c for c in FEAT_COLS if c not in df.columns]
    if miss:
        raise RuntimeError(f"{os.path.basename(path)} -> missing column: {miss[:5]}")
    
    df = df.dropna(subset=[FEAT_COLS[0]]).reset_index(drop=True)
    df[FEAT_COLS] = df[FEAT_COLS].astype('float32')
    return df


def csv_to_windows(path: str, file_id: int):
    """Convert CSV file to windowed data"""
    label = infer_label_from_name(path)
    if label is None:
        print(f" missing label -> {os.path.basename(path)}")
        return None
    
    df = read_csv_safe(path)
    x = df[FEAT_COLS].to_numpy(np.float32)
    
    # Z-normalization
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    
    X, y, fidx = [], [], []
    for s in range(0, len(x) - WIN + 1, STRIDE):
        X.append(x[s:s+WIN])
        y.append(LBL2ID[label])
        fidx.append(file_id)
    
    if not X:
        print(f" missing windows -> {os.path.basename(path)}")
        return None
    
    return np.stack(X), np.array(y, np.int64), np.array(fidx, np.int64)


def find_all_csvs(folder):
    """Find all CSV files for all classes"""
    pats = ["Alerted-*.csv", "Concentrated-*.csv", "Drowsy-*.csv", 
            "Neutral-*.csv", "Relaxed-*.csv"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(folder, p))
    files = sorted(files)
    
    if not files:
        raise SystemExit(f"CSV not found: {folder}")
    
    print(f"[{folder}] {len(files)} files")
    for f in files:
        print("*", os.path.basename(f))
    
    return files


def partition_data_non_iid(data_dir, client_id, num_clients=3):
    """
    Partition data in a non-IID heterogeneous way
    
    Args:
        data_dir: Directory containing the CSV files
        client_id: ID of this client (0, 1, 2, ...)
        num_clients: Total number of clients
        
    Returns:
        X_train, y_train: Training data for this client
    """
    print(f"\n[Client {client_id}] Loading and partitioning data...")
    print(f"[Client {client_id}] Data directory: {data_dir}")
    
    # Load all files
    all_files = find_all_csvs(data_dir)
    
    # Convert all CSVs to windows
    X_list, y_list, idx_list = [], [], []
    for fid, f in enumerate(all_files):
        out = csv_to_windows(f, fid)
        if out is None:
            continue
        Xi, yi, fi = out
        X_list.append(Xi)
        y_list.append(yi)
        idx_list.append(fi)
    
    X_all = np.concatenate(X_list, axis=0).astype("float32")
    y_all = np.concatenate(y_list, axis=0).astype("int64")
    findex = np.concatenate(idx_list, axis=0).astype("int64")
    
    print(f"\n[Client {client_id}] Total windows: {len(y_all)} | Shape: {X_all.shape}")
    
    # Group files by class
    file_to_cls = {fid: infer_label_from_name(all_files[fid]) 
                   for fid in range(len(all_files))}
    
    cls_to_files = {c: [fid for fid in range(len(all_files)) 
                        if file_to_cls[fid] == c] 
                   for c in CLASS_ORDER}
    
    # Shuffle files within each class
    for c in CLASS_ORDER:
        random.shuffle(cls_to_files[c])
    
    # Non-IID distribution strategy:
    # - Assign files to clients such that each client gets different class distributions
    # - Some clients may be missing certain classes entirely
    client_files = {i: [] for i in range(num_clients)}
    
    # Strategy: Give each client at least one file from each class, then distribute rest unevenly
    for c in CLASS_ORDER:
        pool = cls_to_files[c]
        if len(pool) < num_clients:
            print(f"[WARNING] Class '{c}' has only {len(pool)} files, "
                  f"but {num_clients} clients exist")
        
        # Give one file to each client if possible
        for i in range(min(num_clients, len(pool))):
            if pool:
                client_files[i].append(pool.pop())
    
    # Distribute remaining files in a skewed manner
    # This creates heterogeneity - some clients get more of certain classes
    already = set(fid for lst in client_files.values() for fid in lst)
    leftover = [fid for fid in range(len(all_files)) if fid not in already]
    random.shuffle(leftover)
    
    # Skewed distribution - give more files to earlier clients for certain classes
    for idx, fid in enumerate(leftover):
        # Bias towards certain clients based on class
        cls = file_to_cls[fid]
        if cls == 'alerted':
            target_client = idx % max(1, num_clients - 1)  # Favor client 0, 1
        elif cls == 'concentrated':
            target_client = (client_id + 1) % num_clients  # Different distribution
        elif cls == 'drowsy':
            target_client = idx % num_clients  # Even distribution
        else:
            target_client = random.randrange(num_clients)  # Random
        
        client_files[target_client].append(fid)
    
    # Extract data for this client
    sel = set(client_files[client_id])
    m = np.isin(findex, list(sel))
    X_client = X_all[m]
    y_client = y_all[m]
    
    # Print distribution
    cls_cnt = {ID2LBL[k]: int(v) for k, v in Counter(y_client.tolist()).items()}
    missing = [c for c in CLASS_ORDER if LBL2ID[c] not in Counter(y_client.tolist())]
    
    print(f"\n[Client {client_id}] Data partition summary:")
    print(f"  Total windows: {len(y_client)}")
    print(f"  Class distribution: {cls_cnt}")
    if missing:
        print(f"  Missing classes: {missing}")
    print(f"  Files assigned: {len(client_files[client_id])}")
    
    return X_client, y_client


def get_client_data(client_id, num_clients=3, data_dir=None):
    """
    Main function to get data for a specific client
    
    Args:
        client_id: ID of this client (0, 1, 2, ...)
        num_clients: Total number of clients
        data_dir: Directory containing CSV files (defaults to ./Dataset)
        
    Returns:
        X_train, y_train: Training data for this client
    """
    if data_dir is None:
        # Default to Dataset folder in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "Dataset")
    
    print(f"\n{'='*70}")
    print(f"EEG Data Partitioner - Client {client_id}")
    print(f"{'='*70}")
    
    X_train, y_train = partition_data_non_iid(data_dir, client_id, num_clients)
    
    print(f"\n[Client {client_id}] Data partitioning complete!")
    print(f"{'='*70}\n")
    
    return X_train, y_train


if __name__ == "__main__":
    # Test the partitioner
    import sys
    
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    X, y = get_client_data(client_id, num_clients)
    print(f"\nTest complete. Client {client_id} has {len(y)} samples.")
    print(f"Data shape: {X.shape}")
