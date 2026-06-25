"""
Stage 2b - Train Content-Based Filtering
==========================================
Menghitung matriks TF-IDF dan Cosine Similarity dari metadata soup
film yang telah dibuat pada tahap preprocess.

Alur:
1. Load data/processed/content_df.csv (berisi soup tiap film)
2. Vektorisasi TF-IDF pada kolom soup
3. Hitung Cosine Similarity Matrix (shape: top_K x top_K)
4. Simpan matriks ke data/processed/cosine_sim.npy

Output:
- data/processed/cosine_sim.npy    : Matriks Cosine Similarity
- data/processed/content_index.csv : Mapping title -> indeks baris matriks
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import os

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

prep_params = params["preprocessing"]

print("=" * 60)
print("STAGE 2b: CONTENT-BASED FILTERING")
print("=" * 60)

# Load data content
print("\n[1/4] Loading content data...")
df = pd.read_csv(prep_params["output_content"])
df["soup"] = df["soup"].fillna("")
print(f"  Total films: {len(df)}")

# TF-IDF Vectorization
print("\n[2/4] Fitting TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["soup"])
print(f"  TF-IDF Matrix shape: {tfidf_matrix.shape}")

# Cosine Similarity
print("\n[3/4] Computing Cosine Similarity Matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"  Cosine Sim Matrix shape: {cosine_sim.shape}")

# Simpan matriks dan indeks
print("\n[4/4] Saving outputs...")
os.makedirs("data/processed", exist_ok=True)
np.save(prep_params["output_cosine_sim"], cosine_sim)
print(f"  Saved cosine sim: {prep_params['output_cosine_sim']}")

# Simpan mapping judul -> indeks
content_index = df[["title"]].copy().drop_duplicates(subset=["title"]).reset_index()
content_index.to_csv("data/processed/content_index.csv", index=False)
print("  Saved content index: data/processed/content_index.csv")

# Demo rekomendasi untuk verifikasi
print("\n  Demo rekomendasi untuk 'The Dark Knight Rises':")
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
if "The Dark Knight Rises" in indices:
    idx = indices["The Dark Knight Rises"]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:6]
    for i, (movie_idx, score) in enumerate(sim_scores, 1):
        print(f"    {i}. {df['title'].iloc[movie_idx]} (score: {score:.4f})")
else:
    print("    Film tidak ada dalam subset top K, coba judul lain.")

print("\n[DONE] Stage 2b Content-Based Filtering selesai.")
