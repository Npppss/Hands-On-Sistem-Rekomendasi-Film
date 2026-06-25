"""
Inference module - menghasilkan rekomendasi dari ketiga model.

Usage:
    python src/recommend.py --method demographic
    python src/recommend.py --method content --title "The Dark Knight Rises"
    python src/recommend.py --method collaborative --user-id 5
    python src/recommend.py --method all --user-id 5 --title "The Dark Knight Rises"
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RecommenderNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_params():
    with open(os.path.join(PROJECT_ROOT, "params.yaml"), "r") as f:
        return yaml.safe_load(f)


def _path(params, key_path):
    """Resolve path relatif terhadap project root."""
    parts = key_path.split(".")
    val = params
    for p in parts:
        val = val[p]
    return os.path.join(PROJECT_ROOT, val)


def _load_mappings(params):
    with open(_path(params, "preprocessing.output_id_mappings")) as f:
        return json.load(f)


def recommend_demographic(params, top_n=10):
    df = pd.read_csv(_path(params, "train_demographic.output_results"))
    return df.head(top_n)


def recommend_content(params, title, top_n=10):
    df = pd.read_csv(_path(params, "preprocessing.output_content"))
    cosine_sim = np.load(_path(params, "preprocessing.output_cosine_sim"))

    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    if title not in indices:
        available = [t for t in indices.index if title.lower() in t.lower()][:5]
        raise ValueError(
            f"Film '{title}' tidak ditemukan. "
            f"Coba judul serupa: {available or 'tidak ada'}"
        )

    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    results = []
    for movie_idx, score in sim_scores:
        results.append({"title": df["title"].iloc[movie_idx], "similarity": round(score, 4)})
    return pd.DataFrame(results)


def recommend_collaborative(params, user_id, top_n=10):
    mappings = _load_mappings(params)
    movies_df = pd.read_csv(_path(params, "preprocessing.output_processed"))
    ratings = pd.read_csv(_path(params, "preprocessing.output_ratings_encoded"))
    model = tf.keras.models.load_model(
        _path(params, "train_collaborative.output_model"),
        custom_objects={"RecommenderNet": RecommenderNet},
    )

    user2idx = mappings["user2idx"]
    movie2idx = mappings["movie2idx"]
    movieId_to_tmdbId = {int(k): int(v) for k, v in mappings["movieId_to_tmdbId"].items()}

    user_key = str(user_id)
    if user_key not in user2idx:
        raise ValueError(f"User ID {user_id} tidak ada dalam dataset ratings")

    user_idx = user2idx[user_key]
    watched_movie_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"].tolist())
    candidate_movie_ids = [
        int(mid) for mid in movie2idx.keys()
        if int(mid) not in watched_movie_ids
    ]

    if not candidate_movie_ids:
        return pd.DataFrame(columns=["title", "predicted_rating", "genres"])

    pairs = np.array([[user_idx, movie2idx[str(mid)]] for mid in candidate_movie_ids])
    predictions = model.predict(pairs, verbose=0).flatten()
    top_indices = predictions.argsort()[-top_n:][::-1]

    results = []
    for i in top_indices:
        movie_id = candidate_movie_ids[i]
        tmdb_id = movieId_to_tmdbId.get(movie_id)
        title, genres = "Unknown", ""
        if tmdb_id is not None:
            match = movies_df.loc[movies_df["id"] == tmdb_id]
            if not match.empty:
                title = match.iloc[0]["title"]
                genres = match.iloc[0]["genres"]

        results.append({
            "title": title,
            "predicted_rating": round(float(predictions[i]), 2),
            "genres": genres,
            "movieId": movie_id,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Sistem Rekomendasi Film - Inference")
    parser.add_argument(
        "--method",
        choices=["demographic", "content", "collaborative", "all"],
        default="all",
    )
    parser.add_argument("--title", default="The Dark Knight Rises")
    parser.add_argument("--user-id", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    params = _load_params()

    if args.method in ("demographic", "all"):
        print("\n=== Top Film (Demographic Filtering) ===")
        print(recommend_demographic(params, args.top_n).to_string(index=False))

    if args.method in ("content", "all"):
        print(f"\n=== Rekomendasi Serupa: '{args.title}' (Content-Based) ===")
        try:
            print(recommend_content(params, args.title, args.top_n).to_string(index=False))
        except ValueError as e:
            print(f"  {e}")

    if args.method in ("collaborative", "all"):
        print(f"\n=== Rekomendasi Personal: User {args.user_id} (Collaborative) ===")
        try:
            print(recommend_collaborative(params, args.user_id, args.top_n).to_string(index=False))
        except ValueError as e:
            print(f"  {e}")


if __name__ == "__main__":
    main()
