"""
Stage 3 - Evaluate
==================
Mengumpulkan metrik evaluasi dari semua model dan menyimpannya
ke dalam file metrics/metrics.json agar dapat dibaca oleh DVC
dan ditampilkan di dashboard Dagshub.

Metrik yang dikumpulkan:
- Demographic Filtering  : jumlah film yang lolos filter (qualified_count)
- Content-Based Filtering: ukuran matriks kemiripan (cosine_matrix_size)
- Collaborative Filtering: final training loss, final validation loss, best val loss
"""

import pandas as pd
import numpy as np
import json
import yaml
import os

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

prep_params = params["preprocessing"]
demo_params = params["train_demographic"]
collab_params = params["train_collaborative"]
eval_params = params["evaluate"]

print("=" * 60)
print("STAGE 3: EVALUATE")
print("=" * 60)

os.makedirs("metrics", exist_ok=True)
metrics = {}

# -------------------------------------------------------------------
# 1. Metrik Demographic Filtering
# -------------------------------------------------------------------
print("\n[1/3] Evaluating Demographic Filtering...")
demo_df = pd.read_csv(demo_params["output_results"])
metrics["demographic"] = {
    "top10_avg_score": round(float(demo_df["score"].mean()), 4),
    "top10_avg_vote_count": round(float(demo_df["vote_count"].mean()), 0),
    "top10_avg_vote_average": round(float(demo_df["vote_average"].mean()), 4),
}
print(f"  Top-10 avg score: {metrics['demographic']['top10_avg_score']}")

# -------------------------------------------------------------------
# 2. Metrik Content-Based Filtering
# -------------------------------------------------------------------
print("\n[2/3] Evaluating Content-Based Filtering...")
cosine_sim = np.load(prep_params["output_cosine_sim"])
n = cosine_sim.shape[0]

# Distribusi skor similarity rata-rata (eksklusif self-match diagonal)
upper_tri = cosine_sim[np.triu_indices(n, k=1)]
metrics["content_based"] = {
    "corpus_size": n,
    "mean_similarity": round(float(upper_tri.mean()), 6),
    "max_similarity": round(float(upper_tri.max()), 6),
}
print(f"  Corpus size: {n}, Mean similarity: {metrics['content_based']['mean_similarity']}")

# -------------------------------------------------------------------
# 3. Metrik Collaborative Filtering
# -------------------------------------------------------------------
print("\n[3/3] Evaluating Collaborative Filtering...")
with open(collab_params["output_history"], "r") as f:
    history = json.load(f)

train_losses = history["loss"]
val_losses = history["val_loss"]
metrics["collaborative"] = {
    "final_train_loss": round(train_losses[-1], 6),
    "final_val_loss": round(val_losses[-1], 6),
    "best_val_loss": round(min(val_losses), 6),
    "final_train_mae": round(history.get("mae", [0])[-1], 6),
    "final_val_mae": round(history.get("val_mae", [0])[-1], 6),
    "epochs_trained": len(train_losses),
}
print(f"  Final train loss: {metrics['collaborative']['final_train_loss']}")
print(f"  Final val loss  : {metrics['collaborative']['final_val_loss']}")
print(f"  Final val MAE   : {metrics['collaborative']['final_val_mae']}")

# -------------------------------------------------------------------
# 4. Simpan Metrics
# -------------------------------------------------------------------
with open(eval_params["output_metrics"], "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Saved all metrics: {eval_params['output_metrics']}")

# Tampilkan ringkasan
print("\n" + "=" * 60)
print("RINGKASAN METRIK:")
print(json.dumps(metrics, indent=2))
print("=" * 60)

print("\n[DONE] Stage 3 Evaluate selesai.")
