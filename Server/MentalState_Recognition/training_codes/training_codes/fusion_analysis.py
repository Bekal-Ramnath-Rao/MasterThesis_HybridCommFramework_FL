"""
Multimodal Model Pretraining - Centralized Training
Train on 12 client data partitions, save model weights for federated learning
"""

import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#===============================================================================
# CONFIGURATION
#===============================================================================
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
DATA_ROOT = "./recording"
WINDOW_DURATION, STEP_DURATION, TRAIN_RATIO = 10.0, 5.0, 0.8

# Features
TELEMETRY_COLS = ["x", "y", "z", "yaw", "pitch", "roll", "speed",
                  "acc_x", "acc_y", "acc_z", "traffic_density",
                  "sun_altitude", "ambient_light", "snow"]
PPG_COLS = ["PPG1", "PPG2", "PPG3"]
EEG_COLS = ["TP9", "AF7", "AF8", "TP10"]
EEG_FS = 256

# Video
TARGET_FPS, FRAME_SIZE = 5, (112, 112)
FRAMES_PER_WINDOW = int(WINDOW_DURATION * TARGET_FPS)

# Pretraining parameters
NUM_CLIENTS = 12
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3
NUM_EPOCHS = 50
PATIENCE = 10

# Save path
SAVE_DIR = "./pretrained_models/test"
os.makedirs(SAVE_DIR, exist_ok=True)

#===============================================================================
# MODEL ARCHITECTURES (Same as your original)
#===============================================================================
class TelemetryBiLSTM(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(128, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.feature_out = nn.Linear(64, 256)

    def forward(self, x, lengths):
        if x is None:
            return None
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm1(packed)
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)
        unpacked = self.dropout1(unpacked)
        packed_in = pack_padded_sequence(unpacked, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.bilstm2(packed_in)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.feature_out(out)
        return self.relu(out)


class PPGEEGEarlyFusionLSTM(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.feature_out = nn.Linear(64, 256)

    def forward(self, x, lengths):
        if x is None:
            return None
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = hidden[-1]
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.feature_out(out)
        return self.relu(out)


class Video3DCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv4 = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(128, 256)
        self.feature_out = nn.Linear(256, 256)

    def forward(self, x):
        if x is None:
            return None
        x = self.relu(self.conv1(x)); x = self.pool(x); x = self.dropout(x)
        x = self.relu(self.conv2(x)); x = self.pool(x); x = self.dropout(x)
        x = self.relu(self.conv3(x)); x = self.pool(x); x = self.dropout(x)
        x = self.relu(self.conv4(x)); x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x)); x = self.dropout(x)
        return self.relu(self.feature_out(x))


class MultiModalFusionModel(nn.Module):
    def __init__(self, telemetry_dim, ppg_eeg_dim, num_classes, dropout=0.3):
        super().__init__()
        self.telemetry_encoder = TelemetryBiLSTM(telemetry_dim, dropout=dropout)
        self.ppg_eeg_encoder = PPGEEGEarlyFusionLSTM(ppg_eeg_dim, dropout=dropout)
        self.video_encoder = Video3DCNN(dropout=dropout)

        self.telemetry_weight = nn.Parameter(torch.tensor(1.0))
        self.ppg_eeg_weight = nn.Parameter(torch.tensor(1.0))
        self.video_weight = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.num_classes = num_classes

    def forward(self, telemetry, tel_lens, ppg_eeg, ppg_lens, video, availability):
        batch_size = len(availability['telemetry'])
        dev = self.fc1.weight.device

        tel_features = torch.zeros(batch_size, 256, device=dev)
        ppg_features = torch.zeros(batch_size, 256, device=dev)
        vid_features = torch.zeros(batch_size, 256, device=dev)

        if telemetry is not None and tel_lens is not None:
            tel_feat = self.telemetry_encoder(telemetry, tel_lens)
            tel_idx = 0
            for i, has_tel in enumerate(availability['telemetry']):
                if has_tel:
                    tel_features[i] = tel_feat[tel_idx]
                    tel_idx += 1

        if ppg_eeg is not None and ppg_lens is not None:
            ppg_feat = self.ppg_eeg_encoder(ppg_eeg, ppg_lens)
            ppg_idx = 0
            for i, has_ppg in enumerate(availability['ppg_eeg']):
                if has_ppg:
                    ppg_features[i] = ppg_feat[ppg_idx]
                    ppg_idx += 1

        if video is not None:
            vid_feat = self.video_encoder(video)
            vid_idx = 0
            for i, has_vid in enumerate(availability['video']):
                if has_vid:
                    vid_features[i] = vid_feat[vid_idx]
                    vid_idx += 1

        combined = torch.cat([tel_features, ppg_features, vid_features], dim=1)
        out = self.fc1(combined); out = self.relu(out); out = self.dropout(out)
        out = self.fc2(out); out = self.relu(out); out = self.dropout(out)
        return self.fc3(out)

#===============================================================================
# DATA LOADING (Your original implementation)
#===============================================================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    if len(data.shape) == 1:
        return filtfilt(b, a, data)
    return filtfilt(b, a, data, axis=0)

def load_telemetry(csv_path, feature_cols):
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns:
        return None, None
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    available = [c for c in feature_cols if c in df.columns]
    df = df[['time'] + available].dropna().reset_index(drop=True)
    df = df.sort_values('time').reset_index(drop=True)
    return df, available

def load_ppg(csv_path):
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns:
        return None, None
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    available = [c for c in PPG_COLS if c in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['time'] + available].dropna().reset_index(drop=True)
    df = df.sort_values('time').reset_index(drop=True)
    return df, available

def load_eeg(csv_path):
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns:
        return None, None
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    available = [c for c in EEG_COLS if c in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['time'] + available].dropna().reset_index(drop=True)
    df = df.sort_values('time').reset_index(drop=True)
    if len(available) > 0:
        df[available] = bandpass_filter(df[available].values, 1, 40, EEG_FS)
    return df, available

def create_multimodal_index(root_dir):
    data_index = []
    print("\n" + "="*80)
    print(" INDEXING MULTIMODAL DATASET")
    print("="*80)
    for participant in sorted(os.listdir(root_dir)):
        p_path = os.path.join(root_dir, participant)
        if not os.path.isdir(p_path):
            continue
        print(f"\n Participant: {participant}")
        for scenario in sorted(os.listdir(p_path)):
            s_path = os.path.join(p_path, scenario)
            if not os.path.isdir(s_path):
                continue
            trial_count = 0
            for trial in sorted(os.listdir(s_path)):
                t_path = os.path.join(s_path, trial)
                tel_path = os.path.join(t_path, "data.csv")
                ppg_path = os.path.join(t_path, "ppg.csv")
                eeg_path = os.path.join(t_path, "eeg.csv")
                video_path = os.path.join(t_path, "face.mp4")
                has_tel = os.path.exists(tel_path)
                has_ppg = os.path.exists(ppg_path)
                has_eeg = os.path.exists(eeg_path)
                has_video = os.path.exists(video_path)
                if sum([has_tel, has_ppg and has_eeg, has_video]) < 2:
                    continue
                data_index.append({
                    'participant': participant, 'scenario': scenario, 'trial': trial,
                    'trial_path': t_path,
                    'has_telemetry': has_tel, 'has_ppg': has_ppg,
                    'has_eeg': has_eeg, 'has_video': has_video,
                    'label': scenario
                })
                trial_count += 1
            if trial_count > 0:
                print(f"   📁 '{scenario}': {trial_count} trials")
    return data_index

class MultiModalDataset(Dataset):
    def __init__(self, data_index, split='train'):
        self.data_index = data_index
        self.split = split
        self.samples = []
        for trial_info in data_index:
            self._index_trial(trial_info)

    def _index_trial(self, trial_info):
        total_duration = 300.0
        num_windows = int((total_duration - WINDOW_DURATION) / STEP_DURATION) + 1
        split_idx = int(TRAIN_RATIO * num_windows)
        for i in range(num_windows):
            is_train = (i < split_idx)
            if (self.split == 'train' and is_train) or (self.split == 'test' and not is_train):
                self.samples.append({
                    'trial_info': trial_info,
                    'window_idx': i,
                    'window_start': i * STEP_DURATION,
                    'window_end': i * STEP_DURATION + WINDOW_DURATION,
                    'participant': trial_info['participant']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ti = sample['trial_info']
        t_path = ti['trial_path']
        ws, we = sample['window_start'], sample['window_end']

        telemetry = self._load_telemetry_window(os.path.join(t_path, "data.csv"), ws, we) if ti['has_telemetry'] else None
        ppg_eeg = self._load_ppg_eeg_window(os.path.join(t_path, "ppg.csv"), os.path.join(t_path, "eeg.csv"), ws, we) \
                    if ti['has_ppg'] and ti['has_eeg'] else None
        video = self._load_video_window(os.path.join(t_path, "face.mp4"), ws, we) if ti['has_video'] else None
        return telemetry, ppg_eeg, video, ti['label']

    def _load_telemetry_window(self, csv_path, s, e):
        try:
            df, feats = load_telemetry(csv_path, TELEMETRY_COLS)
            if df is None: return None
            mask = (df['time'] >= s) & (df['time'] < e)
            data = df.loc[mask, feats].values
            return torch.FloatTensor(data) if len(data) >= 10 else None
        except:
            return None

    def _load_ppg_eeg_window(self, ppg_path, eeg_path, s, e):
        try:
            df_ppg, ppg_cols = load_ppg(ppg_path)
            df_eeg, eeg_cols = load_eeg(eeg_path)
            if df_ppg is None or df_eeg is None: return None
            ppg_data = df_ppg.loc[(df_ppg['time'] >= s) & (df_ppg['time'] < e), ppg_cols].values
            eeg_data = df_eeg.loc[(df_eeg['time'] >= s) & (df_eeg['time'] < e), eeg_cols].values
            if len(ppg_data) < 10 or len(eeg_data) < 10: return None
            n = min(len(ppg_data), len(eeg_data))
            return torch.FloatTensor(np.concatenate([ppg_data[:n], eeg_data[:n]], axis=1))
        except:
            return None

    def _load_video_window(self, video_path, s, e):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            stride = max(int(fps / TARGET_FPS), 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
            frames, cur = [], int(s * fps)
            while cur < int(e * fps):
                ret, frame = cap.read()
                if not ret: break
                if (cur - int(s * fps)) % stride == 0:
                    frame = cv2.cvtColor(cv2.resize(frame, FRAME_SIZE), cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cur += 1
            cap.release()
            if len(frames) < 10: return None
            frames = self._pad_or_truncate(frames, FRAMES_PER_WINDOW)
            t = np.stack(frames, 0).astype(np.float32) / 255.0
            return torch.FloatTensor(t).permute(3, 0, 1, 2)
        except:
            return None

    def _pad_or_truncate(self, frames, target):
        if len(frames) > target: return frames[:target]
        return frames + [frames[-1]] * (target - len(frames))

def collate_multimodal(batch):
    tel_list, ppg_list, vid_list, labels = zip(*batch)

    tel_valid = [t for t in tel_list if t is not None]
    if tel_valid:
        tel_pad = pad_sequence(tel_valid, batch_first=True, padding_value=0.0)
        tel_lens = torch.LongTensor([len(t) for t in tel_valid])
    else:
        tel_pad, tel_lens = None, None

    ppg_valid = [p for p in ppg_list if p is not None]
    if ppg_valid:
        ppg_pad = pad_sequence(ppg_valid, batch_first=True, padding_value=0.0)
        ppg_lens = torch.LongTensor([len(p) for p in ppg_valid])
    else:
        ppg_pad, ppg_lens = None, None

    vid_valid = [v for v in vid_list if v is not None]
    vid_stack = torch.stack(vid_valid, 0) if vid_valid else None

    availability = {
        'telemetry': [t is not None for t in tel_list],
        'ppg_eeg': [p is not None for p in ppg_list],
        'video': [v is not None for v in vid_list],
    }
    return tel_pad, tel_lens, ppg_pad, ppg_lens, vid_stack, availability, labels

#===============================================================================
# TRAINING AND EVALUATION
#===============================================================================
def train_epoch(model, train_loader, criterion, optimizer, label_encoder, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for tel, tel_lens, ppg, ppg_lens, vid, avail, labels_str in tqdm(train_loader, desc='Training', leave=False):
        if tel is not None:
            tel, tel_lens = tel.to(device), tel_lens.to(device)
        if ppg is not None:
            ppg, ppg_lens = ppg.to(device), ppg_lens.to(device)
        if vid is not None:
            vid = vid.to(device)

        labels = torch.LongTensor([label_encoder.transform([lbl])[0] for lbl in labels_str]).to(device)

        optimizer.zero_grad()
        outputs = model(tel, tel_lens, ppg, ppg_lens, vid, avail)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total

def evaluate(model, test_loader, label_encoder, device):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for tel, tel_lens, ppg, ppg_lens, vid, avail, labels_str in tqdm(test_loader, desc='Evaluating', leave=False):
            if tel is not None:
                tel, tel_lens = tel.to(device), tel_lens.to(device)
            if ppg is not None:
                ppg, ppg_lens = ppg.to(device), ppg_lens.to(device)
            if vid is not None:
                vid = vid.to(device)

            labels = torch.LongTensor([label_encoder.transform([lbl])[0] for lbl in labels_str])
            outputs = model(tel, tel_lens, ppg, ppg_lens, vid, avail)
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='macro')

    return acc, f1, preds, labels_all

#===============================================================================
# RESULTS SAVING HELPER  ← NEW
#===============================================================================
def save_results(history, test_acc, test_f1, preds, labels, label_encoder_classes,
                 total_time, best_epoch, save_dir):
    """Save training curves, metrics CSV, and summary text."""

    epochs_ran = len(history['train_loss'])
    epoch_axis = list(range(1, epochs_ran + 1))

    # --- Accuracy curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_axis, [a * 100 for a in history['train_acc']], label='Train Accuracy', linewidth=2)
    ax.plot(epoch_axis, [a * 100 for a in history['test_acc']], label='Val Accuracy', linewidth=2)
    ax.set_title(f'Multimodal — Validation Accuracy\nTotal Time: {total_time} | Best Val Acc: {max(history["test_acc"])*100:.2f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, 'pretrain_val_accuracy.png')
    plt.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"💾 Saved: {acc_path}")

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_axis, history['train_loss'], label='Train Loss', linewidth=2)
    ax.set_title(f'Multimodal — Training Loss\nTotal Time: {total_time}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'pretrain_loss_curve.png')
    plt.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"💾 Saved: {loss_path}")

    # --- F1 curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_axis, history['test_f1'], label='Val F1 (macro)', linewidth=2, color='green')
    ax.set_title(f'Multimodal — Validation F1 Score\nBest F1: {max(history["test_f1"]):.4f} at epoch {best_epoch+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    f1_path = os.path.join(save_dir, 'pretrain_f1_curve.png')
    plt.savefig(f1_path, dpi=150)
    plt.close(fig)
    print(f"💾 Saved: {f1_path}")

    # --- Per-epoch metrics CSV ---
    metrics_df = pd.DataFrame({
        'epoch': epoch_axis,
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'test_acc': history['test_acc'],
        'test_f1': history['test_f1'],
    })
    csv_path = os.path.join(save_dir, 'pretrain_training_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"💾 Saved: {csv_path}")

    # --- Classification report CSV ---
    from sklearn.metrics import classification_report
    report_dict = classification_report(labels, preds, target_names=label_encoder_classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(save_dir, 'pretrain_classification_report.csv')
    report_df.to_csv(report_path)
    print(f"💾 Saved: {report_path}")

    # --- Summary text ---
    summary_path = os.path.join(save_dir, 'pretrain_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MULTIMODAL PRETRAINING SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Total Training Time   : {total_time}\n")
        f.write(f"Epochs Run            : {epochs_ran}\n")
        f.write(f"Best Epoch (F1)       : {best_epoch + 1}\n")
        f.write(f"Best Val F1 (macro)   : {max(history['test_f1']):.4f}\n")
        f.write(f"Final Test Accuracy   : {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"Final Test F1 (macro) : {test_f1:.4f}\n")
        f.write("="*60 + "\n\n")
        f.write("PER-CLASS REPORT:\n")
        f.write(classification_report(labels, preds, target_names=label_encoder_classes))
    print(f"💾 Saved: {summary_path}")

#===============================================================================
# MAIN PRETRAINING
#===============================================================================
def main():
    print("\n" + "="*80)
    print(" MULTIMODAL MODEL PRETRAINING")
    print(f"   Centralized Training with {NUM_CLIENTS} Client Partitions")
    print("="*80)

    data_index = create_multimodal_index(DATA_ROOT)
    if not data_index:
        print(" No data found!")
        return

    print(f"\n Total trials indexed: {len(data_index)}")

    all_labels = [trial['label'] for trial in data_index]
    unique_labels = sorted(set(all_labels))
    label_counts = Counter(all_labels)

    print(f"\nDetected Classes ({len(unique_labels)}):")
    for label in unique_labels:
        print(f"   - {label}: {label_counts[label]} trials")

    train_dataset = MultiModalDataset(data_index, 'train')
    test_dataset = MultiModalDataset(data_index, 'test')

    label_encoder = LabelEncoder().fit(all_labels)
    num_classes = len(label_encoder.classes_)

    print(f"\nDataset Split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    print(f"   Classes: {num_classes}")

    telemetry_dim = len(TELEMETRY_COLS)
    ppg_eeg_dim = len(PPG_COLS) + len(EEG_COLS)

    model = MultiModalFusionModel(telemetry_dim, ppg_eeg_dim, num_classes, DROPOUT_RATE).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_multimodal, num_workers=4)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_multimodal, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n Starting Training ({NUM_EPOCHS} epochs max, patience={PATIENCE})")
    print("="*80)

    best_f1 = 0.0
    best_epoch = 0                          # ← NEW
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_f1': []}

    save_path = os.path.join(SAVE_DIR, 'Final_pretrained_multimodal_best_.pth')

    total_start = time.time()               # ← NEW

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 80)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, label_encoder, device)
        test_acc, test_f1, _, _ = evaluate(model, test_loader, label_encoder, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch              # ← NEW
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'num_classes': num_classes,
                'telemetry_dim': telemetry_dim,
                'ppg_eeg_dim': ppg_eeg_dim,
                'label_encoder_classes': label_encoder.classes_.tolist(),
            }, save_path)

            print(f"✅ New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break

    # ---- Total time ----                  ← NEW
    total_secs = time.time() - total_start
    total_time_str = f"{int(total_secs // 60)}m {int(total_secs % 60)}s"
    print(f"\n⏱️  Total Training Time: {total_time_str}")

    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, test_f1, preds, labels = evaluate(model, test_loader, label_encoder, device)

    print(f"\nBest Model (Epoch {checkpoint['epoch']+1}):")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1-Score: {test_f1:.4f}")

    print("\n" + "="*80)
    print(" CLASSIFICATION REPORT")
    print("="*80)

    target_names = checkpoint['label_encoder_classes']
    print(classification_report(labels, preds, target_names=target_names))

    # ---- Save all results ----            ← NEW
    save_results(history, test_acc, test_f1, preds, labels,
                 target_names, total_time_str, best_epoch, SAVE_DIR)

    print("\n" + "="*80)
    print(" PRETRAINING COMPLETE")
    print("="*80)
    print(f"\n Model saved to: {save_path}")
    print(f"\nTo use in federated learning:")
    print(f"  checkpoint = torch.load('{save_path}', weights_only=False)")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
