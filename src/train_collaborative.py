"""
Stage 2c - Train Collaborative Filtering (Keras Embedding)
===========================================================
Melatih model Neural Network RecommenderNet berbasis Keras Embedding
untuk memprediksi rating film yang akan diberikan oleh user tertentu.

Arsitektur Model:
- User Embedding (num_users x embedding_size)
- Movie Embedding (num_movies x embedding_size)
- User Bias & Movie Bias
- Dot Product (tanpa sigmoid, loss MSE pada skala rating 1-5)

Output:
- models/collaborative_model/          : Model Keras yang tersimpan
- data/results/training_history.json   : Riwayat loss tiap epoch
- data/results/loss_curve.png          : Grafik kurva loss training vs validation
"""

import pandas as pd
import numpy as np
import json
import yaml
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import RecommenderNet

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

prep_params = params["preprocessing"]
collab_params = params["train_collaborative"]

print("=" * 60)
print("STAGE 2c: COLLABORATIVE FILTERING (KERAS EMBEDDING)")
print("=" * 60)
print(f"  TensorFlow version: {tf.__version__}")

# ---------------------------------------------------------------
# MLflow tracking (remote Dagshub jika tersedia, fallback lokal)
# ---------------------------------------------------------------
mlflow_active = False
try:
    import mlflow

    try:
        import dagshub
        dagshub.init(
            repo_owner="Npppss",
            repo_name="Hands-On-Sistem-Rekomendasi-Film",
            mlflow=True,
        )
        print("  MLflow: connected to Dagshub remote")
    except Exception as e:
        mlflow.set_tracking_uri("file:./mlruns")
        print(f"  MLflow: using local tracking (Dagshub unavailable: {e})")

    mlflow.set_experiment("collaborative-filtering")
    mlflow_active = True
except ImportError:
    print("  MLflow not installed, skipping experiment tracking")

# ---------------------------------------------------------------
# 1. Load Data Ratings Encoded
# ---------------------------------------------------------------
print("\n[1/5] Loading encoded ratings data...")
ratings = pd.read_csv(prep_params["output_ratings_encoded"])
num_users = ratings["user"].nunique()
num_movies = ratings["movie"].nunique()
print(f"  Users: {num_users}, Movies: {num_movies}, Ratings: {len(ratings)}")

# ---------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------
print("\n[2/5] Preprocessing data for training...")
x = ratings[["user", "movie"]].values
y = ratings["rating"].values.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=collab_params["test_size"],
    random_state=collab_params["random_state"]
)
print(f"  Train: {x_train.shape}, Val: {x_val.shape}")
print(f"  Rating range: {y.min():.1f} - {y.max():.1f}")

# ---------------------------------------------------------------
# 3. Bangun Model RecommenderNet
# ---------------------------------------------------------------
print("\n[3/5] Building RecommenderNet model...")
embedding_size = collab_params["embedding_size"]

model = RecommenderNet(num_users, num_movies, embedding_size)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=collab_params["learning_rate"]),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
)
print(f"  Embedding size: {embedding_size}, Loss: MSE")

# ---------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------
print("\n[4/5] Training model...")

def _train():
    return model.fit(
        x=x_train, y=y_train,
        batch_size=collab_params["batch_size"],
        epochs=collab_params["epochs"],
        verbose=1,
        validation_data=(x_val, y_val),
    )

if mlflow_active:
    import mlflow

    mlflow.tensorflow.autolog(log_models=False)
    with mlflow.start_run(run_name="recommender-net"):
        mlflow.log_params({
            "embedding_size": embedding_size,
            "learning_rate": collab_params["learning_rate"],
            "batch_size": collab_params["batch_size"],
            "epochs": collab_params["epochs"],
            "num_users": num_users,
            "num_movies": num_movies,
        })
        history = _train()
        mlflow.log_metric("final_val_mae", float(history.history["val_mae"][-1]))
else:
    history = _train()

# ---------------------------------------------------------------
# 5. Simpan Model & Artefak
# ---------------------------------------------------------------
print("\n[5/5] Saving model and artifacts...")
os.makedirs("models", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

model.save(collab_params["output_model"])
print(f"  Saved model: {collab_params['output_model']}")

history_data = {
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "mae": history.history["mae"],
    "val_mae": history.history["val_mae"],
    "min_rating": float(y.min()),
    "max_rating": float(y.max()),
}
with open(collab_params["output_history"], "w") as f:
    json.dump(history_data, f, indent=2)
print(f"  Saved history: {collab_params['output_history']}")

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss (MSE)", marker="o")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)", marker="s")
plt.title("Kurva Loss Model Collaborative Filtering")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/results/loss_curve.png", dpi=100)
plt.close()
print("  Saved loss curve: data/results/loss_curve.png")

print("\n[DONE] Stage 2c Collaborative Filtering selesai.")
