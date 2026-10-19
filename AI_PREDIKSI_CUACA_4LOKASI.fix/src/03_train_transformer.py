from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Reshape,
    Embedding,
    RepeatVector,
    Concatenate,
    Flatten
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    exist_ok=True
)


# =====================================================
# LOAD DATA DASARIAN (4 LOKASI)
# =====================================================

print("Loading dataset dasarian 4 lokasi...")


X_seq = np.load(DATA_DIR / "X_seq.npy")   # (N, 36, 9) -> 5 cuaca + 4 waktu siklikal
X_loc = np.load(DATA_DIR / "X_loc.npy")   # (N,)       -> id lokasi 0..3
y = np.load(DATA_DIR / "y_train.npy")     # (N, 18, 5)

meta = pd.read_csv(DATA_DIR / "window_meta.csv")  # baris ke-i -> lokasi window ke-i


print("==========================")
print("DATA SHAPE")
print("==========================")

print("X_seq shape:", X_seq.shape)
print("X_loc shape:", X_loc.shape)
print("y shape    :", y.shape)

N_LOCATIONS = int(X_loc.max()) + 1
print("Jumlah lokasi terdeteksi:", N_LOCATIONS)


# =====================================================
# TRAIN / VALIDATION SPLIT (TIME-BASED, PER LOKASI)
# =====================================================
# PENTING (beda dari versi 1 lokasi): versi lama melakukan split 80/20 dengan
# potong index tunggal (split = int(len(X)*0.8)). Itu TIDAK BOLEH dipakai di
# sini karena window dari 4 lokasi ditumpuk berurutan (semua window Pamekasan
# dulu, baru Sampang, dst) -> potong index tunggal bisa mencampur lokasi yang
# salah ke val/train. Solusinya: split 80/20 dilakukan PER LOKASI (bagian
# akhir waktu tiap lokasi -> validasi), baru digabung.

train_idx = []
val_idx = []

for lokasi in meta["lokasi"].unique():
    idx_lokasi = meta.index[meta["lokasi"] == lokasi].to_numpy()
    # idx_lokasi sudah terurut waktu (window dibangun berurutan per lokasi
    # di 02_preprocessing.py)
    split = int(len(idx_lokasi) * 0.8)
    train_idx.extend(idx_lokasi[:split])
    val_idx.extend(idx_lokasi[split:])

train_idx = np.array(sorted(train_idx))
val_idx = np.array(sorted(val_idx))

X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
X_loc_train, X_loc_val = X_loc[train_idx], X_loc[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("\nJumlah data train:", len(train_idx))
print("Jumlah data val  :", len(val_idx))
print("\nSebaran lokasi di train:")
print(meta.loc[train_idx, "lokasi"].value_counts())
print("\nSebaran lokasi di val:")
print(meta.loc[val_idx, "lokasi"].value_counts())


# =====================================================
# TRANSFORMER BLOCK (SAMA DENGAN VERSI 1 LOKASI)
# =====================================================

def transformer_block(inputs):

    attention = MultiHeadAttention(
        num_heads=2,
        key_dim=32
    )(
        inputs,
        inputs
    )

    attention = Dropout(0.1)(
        attention
    )

    x = LayerNormalization()(
        inputs + attention
    )

    ff = Dense(
        64,
        activation="relu"
    )(x)

    ff = Dense(
        x.shape[-1]
    )(ff)

    return LayerNormalization()(
        x + ff
    )


# =====================================================
# MODEL TRANSFORMER DASARIAN, 4 LOKASI (LOCATION EMBEDDING)
# =====================================================

WINDOW = X_seq.shape[1]        # 36
N_SEQ_FEATURES = X_seq.shape[2]  # 9 (5 cuaca + 4 waktu siklikal)
FORECAST = y.shape[1]           # 18
N_TARGET_FEATURES = y.shape[2]  # 5

LOC_EMBED_DIM = 8   # ukuran embedding lokasi (bebas, kecil karena cuma 4 lokasi)


# Input 1: 36 dasarian terakhir, 9 fitur (5 cuaca + 4 waktu siklikal)
seq_input = Input(
    shape=(WINDOW, N_SEQ_FEATURES),
    name="seq_input"
)

# Input 2: id lokasi (0..3)
loc_input = Input(
    shape=(1,),
    name="loc_input"
)

# Embedding lokasi -> (1, LOC_EMBED_DIM) -> diulang di setiap 36 langkah
# waktu, sehingga transformer "tahu" sedang memproses lokasi mana di setiap
# timestep (bukan cuma di akhir).
loc_embed = Embedding(
    input_dim=N_LOCATIONS,
    output_dim=LOC_EMBED_DIM,
    name="lokasi_embedding"
)(loc_input)                      # (batch, 1, LOC_EMBED_DIM)

loc_embed = Flatten()(loc_embed)  # (batch, LOC_EMBED_DIM)
loc_embed_repeated = RepeatVector(WINDOW)(loc_embed)  # (batch, 36, LOC_EMBED_DIM)

# Gabungkan fitur cuaca+waktu dengan embedding lokasi di setiap timestep
x = Concatenate(axis=-1)([seq_input, loc_embed_repeated])
# shape sekarang: (batch, 36, 9 + LOC_EMBED_DIM)

x = transformer_block(x)

x = GlobalAveragePooling1D()(x)

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(
    0.2
)(x)

# Output:
# 18 dasarian kedepan
# 5 variabel cuaca
outputs = Dense(
    FORECAST * N_TARGET_FEATURES
)(x)

outputs = Reshape(
    (FORECAST, N_TARGET_FEATURES)
)(outputs)


model = Model(
    inputs=[seq_input, loc_input],
    outputs=outputs
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=[
        "mae"
    ]
)


model.summary()


# =====================================================
# CALLBACK
# =====================================================

checkpoint = ModelCheckpoint(
    MODEL_DIR / "transformer_weather_model.keras",
    save_best_only=True,
    monitor="val_loss"
)

early_stop = EarlyStopping(
    patience=15,
    restore_best_weights=True
)


# =====================================================
# TRAINING
# =====================================================

history = model.fit(
    [X_seq_train, X_loc_train],
    y_train,
    validation_data=(
        [X_seq_val, X_loc_val],
        y_val
    ),
    epochs=100,
    batch_size=8,
    callbacks=[
        checkpoint,
        early_stop
    ]
)


print("==========================")
print("TRAINING DASARIAN 4 LOKASI SELESAI")
print("==========================")

print("Model tersimpan:")
print(MODEL_DIR / "transformer_weather_model.keras")
