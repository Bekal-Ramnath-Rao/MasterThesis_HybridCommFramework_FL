#  Federated Averaging
#  CNN+BiLSTM+MHA + tf.data augmentations
#  + round-wise JSON LOG (time+metrics)

import os, glob, warnings, random, time, json  
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------- GPU & Mixed Precision ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("[INFO] GPUs:", tf.config.list_physical_devices('GPU'))
else:
    print("[WARN] GPU not found, running on CPU")

from tensorflow.keras import mixed_precision
try:
    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)
    print("[INFO] Mixed precision + XLA enabled")
except Exception as e:
    print("[INFO] Mixed precision/XLA disabled:", e)

AUTOTUNE = tf.data.AUTOTUNE

# ---------------- Global Settings ----------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR  = "EEG DATASET"  # Folder direction
FS        = 256
WIN_S     = 1.0
WIN       = int(FS*WIN_S)          
STRIDE    = 128                     # 0.5 s
TEST_FRAC = 0.20

# Federated
CLIENTS      = 3
ROUNDS       = 16          
LOCAL_EPOCHS = 5
BATCH        = 256
VAL_SPLIT    = 0.10
PUBLIC_MIX   = 0.0        # Currently OFF state

# Band-power columns
BANDS = ['Delta','Theta','Alpha','Beta','Gamma']
CHANS = ['TP9','AF7','AF8','TP10']
FEAT_COLS = [f'{b}_{c}' for b in BANDS for c in CHANS]  # 20

# Labels
CLASS_ORDER = ['alerted','concentrated','drowsy','neutral','relaxed']
LBL2ID = {c:i for i,c in enumerate(CLASS_ORDER)}
ID2LBL = {i:c for c,i in LBL2ID.items()}
NUM_CLASSES = len(CLASS_ORDER)

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.copy_on_write = True

# ---------------- CSV to Windows ----------------
def infer_label_from_name(path:str):
    n = os.path.basename(path).lower()
    if n.startswith("alerted"): return "alerted"
    if n.startswith("concentrated"): return "concentrated"
    if n.startswith("drowsy"): return "drowsy"
    if n.startswith("neutral"): return "neutral"
    if n.startswith("relaxed"): return "relaxed"
    return None

def read_csv_safe(path:str) -> pd.DataFrame:
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

def csv_to_windows(path:str, file_id:int):
    label = infer_label_from_name(path)
    if label is None:
        print(f" missing label -> {os.path.basename(path)}"); return None
    df = read_csv_safe(path)
    x = df[FEAT_COLS].to_numpy(np.float32)
    # z-norm
    mu = x.mean(axis=0, keepdims=True); sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd

    X, y, fidx = [], [], []
    for s in range(0, len(x) - WIN + 1, STRIDE):
        X.append(x[s:s+WIN]); y.append(LBL2ID[label]); fidx.append(file_id)
    if not X:
        print(f" missing windows -> {os.path.basename(path)}"); return None
    return np.stack(X), np.array(y, np.int64), np.array(fidx, np.int64)

def find_all_csvs(folder):
    pats = ["Alerted-*.csv","Concentrated-*.csv","Drowsy-*.csv","Neutral-*.csv","Relaxed-*.csv"]
    files = []
    for p in pats: files += glob.glob(os.path.join(folder, p))
    files = sorted(files)
    if not files: raise SystemExit(f"CSV not found: {folder}")
    print(f"[{folder}] {len(files)} files")
    for f in files: print("*", os.path.basename(f))
    return files

# ---------------- Load all ----------------
all_files = find_all_csvs(DATA_DIR)
X_list, y_list, idx_list = [], [], []
for fid, f in enumerate(all_files):
    out = csv_to_windows(f, fid)
    if out is None: continue
    Xi, yi, fi = out
    X_list.append(Xi); y_list.append(yi); idx_list.append(fi)

X_all  = np.concatenate(X_list, axis=0).astype("float32")  # (N, 256, 20)
y_all  = np.concatenate(y_list, axis=0).astype("int64")
findex = np.concatenate(idx_list, axis=0).astype("int64")
print(f"\nTotal windows: {len(y_all)} | Input shape: {X_all.shape[1:]}")

# ---------------- Train/Test split ----------------
X_tr_all, X_te, y_tr_all, y_te, f_tr_all, f_te = train_test_split(
    X_all, y_all, findex, test_size=TEST_FRAC, random_state=SEED, stratify=y_all
)
print(f"TRAIN/TEST: {len(y_tr_all)}/{len(y_te)} ({len(y_te)/len(y_all):.2%} test)")
print("TEST classes:", {ID2LBL[k]: int(v) for k,v in Counter(y_te.tolist()).items()})

# ---------------- Non-IID client split  ----------------
train_file_ids = sorted(set(f_tr_all.tolist()))

file_to_cls = {fid: infer_label_from_name(all_files[fid]) for fid in train_file_ids}

cls_to_files = {c: [fid for fid in train_file_ids if file_to_cls[fid] == c] for c in CLASS_ORDER}
for c in CLASS_ORDER:
    random.shuffle(cls_to_files[c])

client_files = {i: [] for i in range(CLIENTS)}
for c in CLASS_ORDER:
    pool = cls_to_files[c]
    if len(pool) < CLIENTS:
        print(f"Class '{c}' has only {len(pool)} train files, "
              f"but {CLIENTS} clients exist")
    for i in range(CLIENTS):
        if pool:
            client_files[i].append(pool.pop())

already = set(fid for lst in client_files.values() for fid in lst)
leftover = [fid for fid in train_file_ids if fid not in already]
random.shuffle(leftover)
for fid in leftover:
    client_files[random.randrange(CLIENTS)].append(fid)

clients_data = {}
print("\n[Client distributions]")
for i in range(CLIENTS):
    sel = set(client_files[i])
    m = np.isin(f_tr_all, list(sel))
    Xc, yc = X_tr_all[m], y_tr_all[m]
    clients_data[i] = (Xc, yc)
    cls_cnt = {ID2LBL[k]: int(v) for k, v in Counter(yc.tolist()).items()}
    missing = [c for c in CLASS_ORDER if LBL2ID[c] not in Counter(yc.tolist())]
    print(f"Client {i+1}: {len(yc)} windows | {cls_cnt}"
          + ("" if not missing else f" | missing: {missing}"))

# ---------------- tf.data ----------------
SMOOTH_EPS = 0.05  # label smoothing

def compute_class_weights(y):
    cc = Counter(y.tolist()); total = max(1, sum(cc.values())); K = NUM_CLASSES
    return {cls: total/(K*cnt) for cls, cnt in cc.items()}

def _time_mask(x, p=0.30, Lmin=8, Lmax=32):
    if tf.random.uniform(()) < p:
        T  = tf.shape(x)[0]
        C  = tf.shape(x)[1]
        L  = tf.random.uniform([], Lmin, Lmax+1, dtype=tf.int32)
        s  = tf.random.uniform([], 0, T-L, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([s, C], x.dtype),
            tf.zeros([L, C], x.dtype),
            tf.ones([T - s - L, C], x.dtype)
        ], axis=0)
        x = x * mask
    return x

def _chan_dropout(x, p=0.30, max_drop=4):
    if tf.random.uniform(()) < p:
        C = tf.shape(x)[1]
        drop_n = tf.minimum(tf.random.uniform([], 1, max_drop+1, dtype=tf.int32), C)
        idx = tf.random.shuffle(tf.range(C))[:drop_n]
        mask = tf.ones([C], x.dtype)
        mask = tf.tensor_scatter_nd_update(mask, tf.reshape(idx, [-1,1]),
                                           tf.zeros([drop_n], x.dtype))
        x = x * mask[tf.newaxis, :]
    return x

def _augment(x):
    # time shift
    shift = tf.random.uniform([], -16, 17, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=0)
    # channel dropout and time mask
    x = _chan_dropout(x)
    x = _time_mask(x)
    # gaussian noise
    x = x + tf.random.normal(tf.shape(x), stddev=0.02, dtype=x.dtype)
    return x

def make_dataset(X, y, batch, training, cw:dict, smooth_eps=SMOOTH_EPS):
    Xtf = tf.convert_to_tensor(X)
    ytf = tf.convert_to_tensor(y)
    sw  = np.array([cw.get(int(k), 1.0) for k in y], dtype=np.float32)
    sww = tf.convert_to_tensor(sw, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((Xtf, ytf, sww))
    if training: ds = ds.shuffle(len(y), seed=SEED, reshuffle_each_iteration=True)

    def _map(x, y, w):
        x = tf.cast(x, tf.float32)
        if training: x = _augment(x)
        y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES, dtype=x.dtype)
        if training and smooth_eps > 0:
            y = (1.0 - smooth_eps) * y + smooth_eps / NUM_CLASSES
        return x, y, w

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

# ---------------- HYBRID Model: Conv-ResNet + Dilations + SE + BiLSTM + MHA ----------------
INPUT_SHAPE = (X_all.shape[1], X_all.shape[2])

def se_block(x, r=8):
    ch = x.shape[-1]
    s  = tf.keras.layers.GlobalAveragePooling1D()(x)
    s  = tf.keras.layers.Dense(max(ch//r, 8), activation='relu')(s)
    s  = tf.keras.layers.Dense(ch, activation='sigmoid', dtype='float32')(s)
    s  = tf.keras.layers.Reshape((1, ch))(s)
    return tf.keras.layers.Multiply()([x, s])

def conv_bn_relu(x, f, k, d=1):
    x = tf.keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def res_block(x, f, k, d=1):
    sc = x
    y  = conv_bn_relu(x, f, k, d)
    y  = tf.keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(y)
    y  = tf.keras.layers.BatchNormalization()(y)
    if sc.shape[-1] != f:
        sc = tf.keras.layers.Conv1D(f, 1, padding="same", use_bias=False)(sc)
        sc = tf.keras.layers.BatchNormalization()(sc)
    y = tf.keras.layers.Add()([y, sc])
    y = tf.keras.layers.ReLU()(y)
    y = se_block(y)  # SE
    return y

def build_model():
    inp = tf.keras.Input(shape=INPUT_SHAPE)  # (256,20)

    x = conv_bn_relu(inp, 64, 7, d=1)
    x = res_block(x, 64, 7, d=1)
    x = tf.keras.layers.MaxPooling1D(2)(x)   # 128

    # Dilated conv stack
    for d in [1, 2, 4]:
        x = res_block(x, 128, 5, d=d)

    # Temporal modeling
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # Self-attention + skip + norm
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x    = tf.keras.layers.Add()([x, attn])
    x    = tf.keras.layers.LayerNormalization()(x)

    # Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inp, out)

    # LR schedule + Adam
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=4, t_mul=2.0, m_mul=0.8, alpha=1e-5
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0)

    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")])
    return model

# ---------------- LOG ----------------
LOG_FL = {
    "round": [], "global_acc": [], "global_top2": [], "global_loss": [],
    "round_train_time": [], "round_eval_time": [],
    "per_client_train_time": [], "round_train_loss": []
}

# ---------------- Federated Training ----------------
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    global_model  = build_model()
    global_weights = global_model.get_weights()

    client_models = [build_model() for _ in range(CLIENTS)]
    client_cw     = [compute_class_weights(clients_data[i][1]) for i in range(CLIENTS)]

    BEST = {"acc": 0.0, "weights": [w.copy() for w in global_weights]}

    for r in range(1, ROUNDS+1):
        print(f"\n[Round {r}/{ROUNDS}]")
        agg_w, agg_n = None, 0

        t_round_train = time.perf_counter()     
        per_client_times = []                   
        weighted_last_losses = []               
        weighted_counts = []                    

        for i in range(CLIENTS):
            Xc, yc = clients_data[i]
            if len(yc) == 0:
                print(f" Client {i+1}: no data"); continue

            local = client_models[i]
            local.set_weights(global_weights)

            ds_tr = make_dataset(Xc, yc, batch=BATCH, training=True,  cw=client_cw[i], smooth_eps=SMOOTH_EPS)

            t0 = time.perf_counter()           
            hist = local.fit(
                ds_tr,
                epochs=LOCAL_EPOCHS,
                verbose=2,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3,
                                                            restore_best_weights=True, verbose=0)]
            )
            per_client_times.append(time.perf_counter() - t0)
            if "loss" in hist.history and len(hist.history["loss"]) > 0:
                weighted_last_losses.append(hist.history["loss"][-1])
                weighted_counts.append(len(yc))

            w = local.get_weights(); n = len(yc)
            agg_w = [wi*n for wi in w] if agg_w is None else [ai + wi*n for ai,wi in zip(agg_w, w)]
            agg_n += n

        t_train = time.perf_counter() - t_round_train 

        # FedAvg
        global_weights = [ai/agg_n for ai in agg_w]
        global_model.set_weights(global_weights)

        # Global eval 
        yte_oh = tf.one_hot(y_te, NUM_CLASSES, dtype=tf.float32)
        ds_te = tf.data.Dataset.from_tensor_slices((X_te.astype("float32"), yte_oh))
        ds_te = ds_te.batch(BATCH).prefetch(AUTOTUNE)

        t_eval0 = time.perf_counter()                 
        loss, acc, top2 = global_model.evaluate(ds_te, verbose=0)
        t_eval = time.perf_counter() - t_eval0         

        print(f"  -> Global TEST acc: {acc:.4f}, top2: {top2:.4f}, loss: {loss:.4f}")
        if acc > BEST["acc"]:
            BEST["acc"] = float(acc)
            BEST["weights"] = [w.copy() for w in global_weights]

        # ---- LOG ----
        LOG_FL["round"].append(r)
        LOG_FL["global_acc"].append(float(acc))
        LOG_FL["global_top2"].append(float(top2))
        LOG_FL["global_loss"].append(float(loss))
        LOG_FL["round_train_time"].append(float(t_train))
        LOG_FL["round_eval_time"].append(float(t_eval))
        LOG_FL["per_client_train_time"].append([float(t) for t in per_client_times])

        if weighted_last_losses:
            w = np.array(weighted_counts, dtype=np.float64)
            v = np.array(weighted_last_losses, dtype=np.float64)
            LOG_FL["round_train_loss"].append(float((v * w).sum() / w.sum()))
        else:
            LOG_FL["round_train_loss"].append(None)

# ---------------- Final Report ----------------
global_model.set_weights(BEST["weights"])
probs = global_model.predict(tf.data.Dataset.from_tensor_slices(X_te.astype("float32")).batch(BATCH),
                             verbose=0)
yp = probs.argmax(axis=1)

print("\n--- Federated (GLOBAL) TEST report ---")
print(classification_report(y_te, yp, target_names=CLASS_ORDER, digits=3, zero_division=0))

cm = confusion_matrix(y_te, yp, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(6.8,4.8))
im = ax.imshow(cm, cmap="Greys")
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_ORDER, rotation=30)
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_ORDER)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (GLOBAL FedAvg)")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color=('white' if cm[i,j] > cm.max()*0.6 else 'black'), fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ---------------- Per-Client Confusion Matrices ----------------


try:
    for i in range(CLIENTS):
        Xc, yc = clients_data[i]
        if len(yc) == 0:
            print(f"[Client {i+1}] no data -> skipped")
            continue

        print(f"[Per-Client CM] Evaluating Client {i+1} (last local model) on GLOBAL test set...")
        local_model = client_models[i]  # local weights

        ds_te_plain = tf.data.Dataset.from_tensor_slices(X_te.astype("float32")).batch(BATCH)
        probs_i = local_model.predict(ds_te_plain, verbose=0)
        yp_i    = probs_i.argmax(axis=1)

        # Confusion matrix
        cm_i = confusion_matrix(y_te, yp_i, labels=list(range(NUM_CLASSES)))

        # Plot
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        im = ax.imshow(cm_i, cmap="Greys")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_ORDER, rotation=30)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Client {i+1} - Last Local Model)")

        for r in range(cm_i.shape[0]):
            for c in range(cm_i.shape[1]):
                ax.text(c, r, str(cm_i[r, c]),
                        ha='center', va='center',
                        color=('white' if cm_i[r, c] > cm_i.max()*0.6 else 'black'),
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.show()
except Exception as e:
    print("[WARN] Per-client confusion matrices could not be generated:", e)


# ---- LOG file ----
if LOG_FL["round"]:
    best_idx = int(np.argmax(LOG_FL["global_acc"]))
    LOG_FL["best"] = {
        "round": int(LOG_FL["round"][best_idx]),
        "acc": float(LOG_FL["global_acc"][best_idx]),
        "loss": float(LOG_FL["global_loss"][best_idx]),
        "top2": float(LOG_FL["global_top2"][best_idx])
    }
with open("metrics_fl.json", "w") as f:
    json.dump(LOG_FL, f, indent=2)
print("[LOG] Saved metrics_fl.json")