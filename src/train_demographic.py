"""
Stage 2a - Train Demographic Filtering
=======================================
Menghitung Weighted Rating IMDB untuk setiap film dan menyimpan
Top 10 film terpopuler ke dalam file results.

Formula:
    W = (v / (v + m)) * R + (m / (v + m)) * C

Di mana:
    v = vote_count film
    m = minimum vote threshold (persentil ke-90)
    R = rata-rata rating film
    C = rata-rata rating seluruh dataset

Output:
- data/results/demographic_results.csv : Top 10 film terpopuler beserta skor weighted rating
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (tidak butuh display)
import matplotlib.pyplot as plt
import yaml
import os

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

prep_params = params["preprocessing"]
demo_params = params["train_demographic"]

print("=" * 60)
print("STAGE 2a: DEMOGRAPHIC FILTERING")
print("=" * 60)

# Load data yang sudah diproses
print("\n[1/4] Loading processed data...")
df = pd.read_csv(prep_params["output_processed"])
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
df = df.dropna(subset=["vote_average", "vote_count"])

# Hitung C dan m
print("\n[2/4] Calculating weighted rating parameters...")
C = df["vote_average"].mean()
vote_pct = prep_params["vote_percentile"]
m = df["vote_count"].quantile(vote_pct)
print(f"  C (mean rating)      : {C:.4f}")
print(f"  m (min votes @p{int(vote_pct*100)}) : {m:.0f}")

# Filter dan hitung weighted rating
print("\n[3/4] Calculating weighted rating scores...")
qualified = df[df["vote_count"] >= m].copy()

def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m)) * R + (m / (m + v)) * C

qualified["score"] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values("score", ascending=False)

top10 = qualified[["title", "vote_count", "vote_average", "score"]].head(10)
print("\nTop 10 Film (Demographic Filtering):")
print(top10.to_string(index=False))

# Simpan hasil
os.makedirs("data/results", exist_ok=True)
qualified[["title", "vote_count", "vote_average", "score"]].head(10).to_csv(
    demo_params["output_results"], index=False
)
print(f"\n[4/4] Saved: {demo_params['output_results']}")

# Simpan plot
print("  Saving chart to data/results/demographic_chart.png...")
plt.figure(figsize=(10, 5))
plt.barh(top10["title"].iloc[::-1], top10["score"].iloc[::-1], color="royalblue")
plt.xlabel("Skor Weighted Rating")
plt.title("Top 10 Film Terpopuler (Demographic Filtering)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("data/results/demographic_chart.png", dpi=100)
plt.close()

print("\n[DONE] Stage 2a Demographic Filtering selesai.")
