"""
Stage 1 - Preprocess
====================
Melakukan pembersihan dan penggabungan data dari empat sumber dataset:
- movies_metadata.csv
- credits.csv
- keywords.csv
- ratings_small.csv
- links_small.csv  (bridge MovieLens movieId <-> TMDB tmdbId)

Output:
- data/processed/processed_movies.csv  : Data film gabungan dengan fitur yang sudah di-parse
- data/processed/content_df.csv        : Data untuk Content-Based Filtering (top K film populer + soup)
- data/processed/ratings_encoded.csv   : Data ratings yang sudah di-encode untuk Collaborative Filtering
- data/processed/id_mappings.json      : Mapping user/movie index dan movieId <-> tmdbId
"""

import pandas as pd
import numpy as np
import ast
import json
import yaml
import os

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data_params = params["data"]
prep_params = params["preprocessing"]

# Buat direktori output jika belum ada
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

print("=" * 60)
print("STAGE 1: PREPROCESS")
print("=" * 60)

# ---------------------------------------------------------------
# 1. Load Raw Data
# ---------------------------------------------------------------
print("\n[1/7] Loading raw datasets...")
movies = pd.read_csv(data_params["movies_path"], low_memory=False)
credits = pd.read_csv(data_params["credits_path"])
keywords = pd.read_csv(data_params["keywords_path"])
ratings = pd.read_csv(data_params["ratings_path"])
links = pd.read_csv(data_params["links_path"])

print(f"  Movies:  {movies.shape}")
print(f"  Credits: {credits.shape}")
print(f"  Keywords:{keywords.shape}")
print(f"  Ratings: {ratings.shape}")
print(f"  Links:   {links.shape}")

# ---------------------------------------------------------------
# 2. Bersihkan dan Merge Dataset Film
# ---------------------------------------------------------------
print("\n[2/7] Cleaning and merging movie datasets...")

movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
movies = movies.dropna(subset=["id"])
movies["id"] = movies["id"].astype("int64")

credits["id"] = credits["id"].astype("int64")
keywords["id"] = keywords["id"].astype("int64")

df = movies.merge(credits, on="id").merge(keywords, on="id")
df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
print(f"  Merged shape: {df.shape}")

# ---------------------------------------------------------------
# 3. Parse kolom JSON
# ---------------------------------------------------------------
print("\n[3/7] Parsing JSON columns...")

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

for col in ["cast", "crew", "keywords", "genres"]:
    df[col] = df[col].apply(safe_literal_eval)

# Ekstrak sutradara dari crew
def get_director(x):
    for i in x:
        if isinstance(i, dict) and i.get("job") == "Director":
            return i.get("name", "")
    return ""

# Ambil top 3 aktor
def get_top3(x):
    if isinstance(x, list):
        return [i["name"] for i in x[:3] if isinstance(i, dict)]
    return []

df["director"] = df["crew"].apply(get_director)
df["cast"] = df["cast"].apply(get_top3)
df["keywords"] = df["keywords"].apply(lambda x: [i["name"] for i in x if isinstance(i, dict)])
df["genres"] = df["genres"].apply(lambda x: [i["name"] for i in x if isinstance(i, dict)])

print("  Columns parsed: director, cast, keywords, genres")

# ---------------------------------------------------------------
# 4. Simpan processed_movies untuk Demographic Stage
# ---------------------------------------------------------------
print("\n[4/7] Saving processed_movies.csv...")
df_save = df[["id", "title", "vote_average", "vote_count", "popularity", "genres"]].copy()
df_save["genres"] = df_save["genres"].apply(lambda x: "|".join(x))
df_save.to_csv(prep_params["output_processed"], index=False)
print(f"  Saved: {prep_params['output_processed']}")

# ---------------------------------------------------------------
# 5. Buat Content Soup untuk Content-Based Filtering
# ---------------------------------------------------------------
print("\n[5/7] Building content soup for Content-Based Filtering...")

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ""

for col in ["cast", "keywords", "director", "genres"]:
    df[col] = df[col].apply(clean_data)

def create_soup(x):
    kw = " ".join(x["keywords"]) if isinstance(x["keywords"], list) else ""
    cast = " ".join(x["cast"]) if isinstance(x["cast"], list) else ""
    genres = " ".join(x["genres"]) if isinstance(x["genres"], list) else ""
    director = x["director"] if isinstance(x["director"], str) else ""
    overview = str(x.get("overview", ""))
    return f"{kw} {cast} {director} {genres} {overview}"

df["soup"] = df.apply(create_soup, axis=1).fillna("")

# Batasi ke top K film populer
top_k = prep_params["content_top_k"]
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
df_content = df.sort_values("popularity", ascending=False).head(top_k).reset_index(drop=True)
df_content[["id", "title", "soup"]].to_csv(prep_params["output_content"], index=False)
print(f"  Saved top {top_k} films: {prep_params['output_content']}")

# ---------------------------------------------------------------
# 6. Encode Ratings untuk Collaborative Filtering
# ---------------------------------------------------------------
print("\n[6/7] Encoding ratings for Collaborative Filtering...")

user_ids = ratings["userId"].unique().tolist()
movie_ids = ratings["movieId"].unique().tolist()

user2idx = {str(x): i for i, x in enumerate(user_ids)}
movie2idx = {str(x): i for i, x in enumerate(movie_ids)}
idx2user = {str(i): x for i, x in enumerate(user_ids)}
idx2movie = {str(i): x for i, x in enumerate(movie_ids)}

ratings["user"] = ratings["userId"].map({x: i for i, x in enumerate(user_ids)})
ratings["movie"] = ratings["movieId"].map({x: i for i, x in enumerate(movie_ids)})

ratings.to_csv(prep_params["output_ratings_encoded"], index=False)
print(f"  Users: {len(user_ids)}, Movies: {len(movie_ids)}")
print(f"  Saved: {prep_params['output_ratings_encoded']}")

# ---------------------------------------------------------------
# 7. Buat mapping MovieLens movieId <-> TMDB tmdbId
# ---------------------------------------------------------------
print("\n[7/7] Building MovieLens <-> TMDB ID mappings...")

links = links.dropna(subset=["movieId", "tmdbId"])
links["movieId"] = links["movieId"].astype(int)
links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)

movieId_to_tmdbId = {str(row.movieId): int(row.tmdbId) for row in links.itertuples()}
tmdbId_to_movieId = {str(row.tmdbId): int(row.movieId) for row in links.itertuples()}

# Hanya film yang ada di ratings DAN punya mapping TMDB
rated_movie_ids = set(movie_ids)
mapped_count = sum(1 for mid in rated_movie_ids if str(mid) in movieId_to_tmdbId)
print(f"  Rated movies with TMDB mapping: {mapped_count}/{len(rated_movie_ids)}")

id_mappings = {
    "user2idx": user2idx,
    "movie2idx": movie2idx,
    "idx2user": idx2user,
    "idx2movie": idx2movie,
    "movieId_to_tmdbId": movieId_to_tmdbId,
    "tmdbId_to_movieId": tmdbId_to_movieId,
}

with open(prep_params["output_id_mappings"], "w") as f:
    json.dump(id_mappings, f, indent=2)
print(f"  Saved: {prep_params['output_id_mappings']}")

print("\n[DONE] Stage 1 Preprocess selesai.")
