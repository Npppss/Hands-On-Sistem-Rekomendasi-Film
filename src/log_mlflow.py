"""
Push metrik eksperimen ke Dagshub MLflow untuk visualisasi di dashboard.
Jalankan setelah pipeline selesai: python src/log_mlflow.py
"""

import json
import os
import yaml

import mlflow

REPO_OWNER = "Npppss"
REPO_NAME = "Hands-On-Sistem-Rekomendasi-Film"

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("params.yaml") as f:
    params = yaml.safe_load(f)

try:
    import dagshub
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    print("MLflow: connected to Dagshub remote")
except Exception as e:
    mlflow.set_tracking_uri("file:./mlruns")
    print(f"MLflow: Dagshub unavailable ({e}), using local tracking")

mlflow.set_experiment("collaborative-filtering")

with open(params["evaluate"]["output_metrics"]) as f:
    metrics = json.load(f)

with open(params["train_collaborative"]["output_history"]) as f:
    history = json.load(f)

collab_params = params["train_collaborative"]

with mlflow.start_run(run_name="pipeline-final"):
    mlflow.log_params({
        "embedding_size": collab_params["embedding_size"],
        "learning_rate": collab_params["learning_rate"],
        "batch_size": collab_params["batch_size"],
        "epochs": collab_params["epochs"],
    })

    for section, section_metrics in metrics.items():
        for key, value in section_metrics.items():
            mlflow.log_metric(f"{section}_{key}", value)

    for i, (loss, val_loss) in enumerate(zip(history["loss"], history["val_loss"])):
        mlflow.log_metric("train_loss", loss, step=i)
        mlflow.log_metric("val_loss", val_loss, step=i)

    for artifact in [
        "data/results/demographic_chart.png",
        "data/results/loss_curve.png",
        params["evaluate"]["output_metrics"],
    ]:
        if os.path.exists(artifact):
            mlflow.log_artifact(artifact)

    print("Logged metrics and artifacts to MLflow")

print("Done. View at: https://dagshub.com/{}/{}/experiments".format(REPO_OWNER, REPO_NAME))
