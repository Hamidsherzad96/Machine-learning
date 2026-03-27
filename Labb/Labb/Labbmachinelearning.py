from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_text(s: str) -> str:
    """Text "rensning" """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class MovieRecommender:
    data_dir: Path

    movies_: Optional[pd.DataFrame] = None
    tfidf_: Optional[sparse.csr_matrix] = None
    vectorizer_: Optional[TfidfVectorizer] = None

    def load(self) -> "MovieRecommender":
        movies_path = self.data_dir / "movies.csv"
        tags_path = self.data_dir / "tags.csv"

        if not movies_path.exists():
            raise FileNotFoundError(f"Missing {movies_path}")
        if not tags_path.exists():
            raise FileNotFoundError(f"Missing {tags_path}")

        movies = pd.read_csv(movies_path)
        tags = pd.read_csv(tags_path)

        # Aggregate tags per movie
        tags["tag_norm"] = tags["tag"].astype(str).map(_normalize_text)

        tag_agg = (
            tags.groupby("movieId")["tag_norm"]
            .apply(lambda x: " ".join(x.tolist()))
            .reset_index()
            .rename(columns={"tag_norm": "tags_text"})
        )

        # Process genres and title
        movies["genres_text"] = (
            movies["genres"]
            .fillna("")
            .astype(str)
            .str.replace("|", " ", regex=False)
            .map(_normalize_text)
        )

        movies["title_text"] = (
            movies["title"]
            .fillna("")
            .astype(str)
            .map(_normalize_text)
        )

        movies = movies.merge(tag_agg, on="movieId", how="left")
        movies["tags_text"] = movies["tags_text"].fillna("")

        # Combine all text features
        movies["combined_text"] = (
            movies["genres_text"] + " "
            + movies["title_text"] + " "
            + movies["tags_text"]
        ).str.strip()

        self.movies_ = movies
        return self

    def fit(self) -> "MovieRecommender":
        if self.movies_ is None:
            raise RuntimeError("Movies not loaded")

        self.vectorizer_ = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=200000,
        )

        self.tfidf_ = self.vectorizer_.fit_transform(self.movies_)
        return self

    def _resolve_movie_index(self, movie: str) -> int:
        titles = self.movies_["title"].astype(str)
        lower = titles.str.lower()
        query = movie.strip().lower()

        exact = self.movies_[lower == query]
        if len(exact) > 0:
            return int(exact.index[0])

        contains = self.movies_[lower.str.contains(query, na=False)]
        if len(contains) > 0:
            return int(contains.index[0])

        raise ValueError(f"Movie {movie} not found")

    def reccomend(self, movie: str, k: int = 5) -> pd.DataFrame:
        if self.movies_ is None or self.tfidf_ is None:
            raise RuntimeError

        idx = self._resolve_movie_index(movie)

        similiarities = cosine_similarity(self.tfidf_[idx], self.tfidf_).ravel()
        similiarities[idx] = -1.0

        top_idx = np.argsort(-similiarities)[:k]

        results = self.movies_.iloc[top_idx].copy()
        results["similarity"] = similiarities[top_idx]

        return results[["movieId", "title", "genres", "similarity"]]

    def main():
        parser = argparse.ArgumentParser(description="Movie recommendation system")
        parser.add_argument("--data", required=True, help="Path to ml-latest folder")
        parser.add_argument("--movie", required=True, help="Movie title (substring allowed)")
        parser.add_argument("--k", type=int, default=5, help="Number of recommendations")

        args = parser.parse_args()

        recommender = MovieRecommender(Path(args.data)).load().fit()
        recommendations = recommender.reccomend(args.movie, k=args.k)

        print(f"\nRecommendations for: {args.movie}\n")
        print(recommendations.to_string(index=False))

    if __name__ == "__main__":
        main()

"""
Skrivs in i terminalen: python LabbMachineLearning.py --data . --movie "Toy Story"

eller andra filmexempel utöver "Toy Story" funkar också bra.
"""














