import logging
import os
import random
from typing import List, Dict, Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_spotify_client() -> Optional[spotipy.Spotify]:
    """
    Initialize and return an authenticated Spotify client using
    Client Credentials flow (no user login required).

    Returns None if credentials are missing or invalid, so callers
    can fall back to the local CSV catalog gracefully.
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        logger.warning("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in .env — Spotify disabled")
        return None

    try:
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        client = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")
        return client
    except Exception as e:
        logger.error("Failed to initialize Spotify client | error=%s", e, exc_info=True)
        return None


def fetch_spotify_recommendations(
    sp: spotipy.Spotify,
    seed_genres: List[str],
    target_energy: float,
    target_valence: float,
    target_danceability: float,
    target_acousticness: float,
    target_tempo: float,
    limit: int = 5,
) -> List[Dict]:
    """
    Search Spotify for tracks by genre and return tracks normalized to the same
    schema as data/songs.csv so score_song() can consume them without modification.

    Note: Spotify's /recommendations endpoint was deprecated for new apps after
    November 2023. This implementation uses /search instead, querying by genre
    and collecting enough candidates for the downstream Gaussian scorer to rank.

    Requires two API calls per genre:
      1. /search        — returns track metadata (id, name, artists, url)
      2. /audio-features — returns audio features (energy, valence, tempo, ...)

    Returns an empty list on any failure so app.py can fall back to songs.csv.

    Schema of each returned dict:
        id, title, artist, genre, mood,
        energy, tempo_bpm, valence, danceability, acousticness, spotify_url
    """
    logger.info(
        "Fetching Spotify recommendations | seed_genres=%s | energy=%.2f | "
        "valence=%.2f | danceability=%.2f | acousticness=%.2f | tempo=%.0f | limit=%d",
        seed_genres, target_energy, target_valence, target_danceability,
        target_acousticness, target_tempo, limit,
    )

    try:
        tracks_per_genre = max(1, limit // len(seed_genres)) if seed_genres else limit
        seen_ids: set = set()
        # Each entry is (track_dict, genre_str)
        all_tracks: List[tuple] = []

        for genre in seed_genres[:5]:
            try:
                response = sp.search(
                    q=f"genre:{genre}",
                    type="track",
                    limit=min(tracks_per_genre + 5, 20),
                )
                if response is None:
                    continue
                items = response.get("tracks", {}).get("items", [])
                new_tracks = 0
                for track in items:
                    track_id = track.get("id")
                    if track_id and track_id not in seen_ids:
                        seen_ids.add(track_id)
                        all_tracks.append((track, genre))
                        new_tracks += 1
                logger.info("Genre search | genre=%r | tracks_found=%d", genre, new_tracks)
            except Exception as e:
                logger.error("Search failed for genre | genre=%r | error=%s", genre, e, exc_info=True)
                continue

        if not all_tracks:
            logger.warning("Spotify search returned 0 tracks across all genres | seed_genres=%s", seed_genres)
            return []

        track_ids = [t["id"] for t, _ in all_tracks if t.get("id")]
        logger.info("Fetching audio features | track_count=%d", len(track_ids))
        features_by_id = _fetch_audio_features_batch(sp, track_ids)
        logger.info(
            "Audio features received | features_count=%d | jitter_fallback_count=%d",
            len(features_by_id),
            len(track_ids) - len(features_by_id),
        )

        results = []
        for track, genre in all_tracks:
            track_id = track.get("id")
            if not track_id:
                continue

            # When /audio-features is unavailable (403 for new Spotify apps),
            # generate per-track jittered values so the scorer has meaningful
            # variance to rank on rather than identical fallback values.
            if track_id in features_by_id:
                features = features_by_id[track_id]
            else:
                logger.warning(
                    "Audio features unavailable, using jittered fallback | track_id=%s | title=%r",
                    track_id, track.get("name", "Unknown"),
                )
                features = {
                    "energy":       max(0.0, min(1.0, target_energy       + random.gauss(0, 0.12))),
                    "valence":      max(0.0, min(1.0, target_valence      + random.gauss(0, 0.12))),
                    "danceability": max(0.0, min(1.0, target_danceability + random.gauss(0, 0.12))),
                    "acousticness": max(0.0, min(1.0, target_acousticness + random.gauss(0, 0.12))),
                    "tempo":        max(40.0, min(220.0, target_tempo     + random.gauss(0, 15))),
                }

            results.append({
                "id":           track_id,
                "title":        track.get("name", "Unknown"),
                "artist":       track["artists"][0]["name"] if track.get("artists") else "Unknown",
                "genre":        genre,
                "mood":         derive_mood(
                                    features.get("valence", target_valence),
                                    features.get("energy", target_energy),
                                ),
                "energy":       features.get("energy", target_energy),
                "tempo_bpm":    features.get("tempo", target_tempo),
                "valence":      features.get("valence", target_valence),
                "danceability": features.get("danceability", target_danceability),
                "acousticness": features.get("acousticness", target_acousticness),
                "spotify_url":  track.get("external_urls", {}).get("spotify", ""),
            })

        logger.info("Spotify recommendations complete | total_returned=%d", len(results[:limit]))
        return results[:limit]

    except Exception as e:
        logger.error("Error fetching Spotify recommendations | error=%s", e, exc_info=True)
        return []


def derive_mood(valence: float, energy: float) -> str:
    """
    Derive a mood label from Spotify audio features using a simple
    valence × energy heuristic.

    Spotify provides no mood field, so this bridges the gap and allows
    score_song()'s mood bonus to fire for live tracks.

    The four quadrants map to moods that Claude's query_parser also produces,
    so the labels are intentionally consistent with those in query_parser.py.

        high valence + high energy → "happy"   (upbeat, positive, energetic)
        low  valence + high energy → "intense"  (aggressive, tense, driving)
        high valence + low  energy → "chill"    (relaxed, easygoing, light)
        low  valence + low  energy → "moody"    (melancholic, dark, reflective)
    """
    if valence >= 0.5 and energy >= 0.5:
        return "happy"
    elif valence < 0.5 and energy >= 0.5:
        return "intense"
    elif valence >= 0.5 and energy < 0.5:
        return "chill"
    else:
        return "moody"


def _fetch_audio_features_batch(
    sp: spotipy.Spotify,
    track_ids: List[str],
) -> Dict[str, Dict]:
    """
    Fetch audio features for a list of track IDs.
    Spotify allows up to 100 IDs per request.

    Returns a dict keyed by track_id → feature dict.
    Returns an empty dict on failure.
    """
    if not track_ids:
        return {}

    try:
        # Spotify enforces a max of 100 IDs per call
        batch = track_ids[:100]
        features_list = sp.audio_features(batch)

        if features_list is None:
            return {}

        result = {}
        for features in features_list:
            if features is None:
                continue
            result[features["id"]] = features

        return result

    except Exception as e:
        logger.error("Error fetching audio features | error=%s", e, exc_info=True)
        return {}
