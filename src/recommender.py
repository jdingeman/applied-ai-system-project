from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":            int(row["id"]),
                "title":         row["title"],
                "artist":        row["artist"],
                "genre":         row["genre"],
                "mood":          row["mood"],
                "energy":        float(row["energy"]),
                "tempo_bpm":     int(row["tempo_bpm"]),
                "valence":       float(row["valence"]),
                "danceability":  float(row["danceability"]),
                "acousticness":  float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """Return a (score, explanation) tuple using Gaussian similarity on numerical features and bonuses for genre/mood matches."""
    import math

    SIGMA = 0.2          # tolerance for numerical features
    BPM_MIN, BPM_MAX = 55, 168  # range from current catalog

    # --- weights must sum to 1.0 ---
    WEIGHTS = {
        "energy":       0.30,
        "valence":      0.25,
        "danceability": 0.20,
        "acousticness": 0.15,
        "tempo":        0.10,
    }

    # Categorical bonus added on top of the weighted numerical score
    GENRE_BONUS = 0.15
    MOOD_BONUS  = 0.10

    def gaussian(diff: float) -> float:
        return math.exp(-(diff ** 2) / (2 * SIGMA ** 2))

    # Normalize tempo to [0, 1] before scoring
    user_tempo_norm = (user_prefs.get("target_bpm", 120) - BPM_MIN) / (BPM_MAX - BPM_MIN)
    song_tempo_norm = (song["tempo_bpm"] - BPM_MIN) / (BPM_MAX - BPM_MIN)

    feature_scores = {
        "energy":       gaussian(abs(user_prefs.get("target_energy", 0.5)       - song["energy"])),
        "valence":      gaussian(abs(user_prefs.get("target_valence", 0.5)      - song["valence"])),
        "danceability": gaussian(abs(user_prefs.get("target_danceability", 0.5) - song["danceability"])),
        "acousticness": gaussian(abs(user_prefs.get("target_acousticness", 0.5) - song["acousticness"])),
        "tempo":        gaussian(abs(user_tempo_norm - song_tempo_norm)),
    }

    numerical_score = sum(WEIGHTS[f] * feature_scores[f] for f in WEIGHTS)

    # Categorical bonuses
    genre_match = song["genre"] == user_prefs.get("favorite_genre", "")
    mood_match  = song["mood"]  == user_prefs.get("favorite_mood", "")
    bonus = (GENRE_BONUS if genre_match else 0) + (MOOD_BONUS if mood_match else 0)

    total_score = min(numerical_score + bonus, 1.0)

    # Build a human-readable explanation
    reasons = []
    if genre_match:
        reasons.append(f"genre matches '{song['genre']}'")
    if mood_match:
        reasons.append(f"mood matches '{song['mood']}'")
    top_feature = max(feature_scores, key=lambda f: feature_scores[f])
    reasons.append(f"strong {top_feature} match ({song[top_feature] if top_feature != 'tempo' else song['tempo_bpm']})")
    explanation = "; ".join(reasons)

    return total_score, explanation

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs against user preferences and return the top k as (song, score, explanation) tuples."""
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
