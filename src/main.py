"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}") 

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    user_pref = {"target_energy": 0.8, "target_bpm": 132, "target_valence": 0.7, "target_danceability": 0.8, "target_acousticness": 0.05 }
    user_pref_pop     = {"favorite_genre": "pop",     "favorite_mood": "happy",   "target_energy": 0.82, "target_bpm": 120, "target_valence": 0.80, "target_danceability": 0.82, "target_acousticness": 0.15}
    user_pref_indie   = {"favorite_genre": "indie pop","favorite_mood": "chill",   "target_energy": 0.55, "target_bpm": 108, "target_valence": 0.70, "target_danceability": 0.60, "target_acousticness": 0.45}
    user_pref_country = {"favorite_genre": "country",  "favorite_mood": "relaxed", "target_energy": 0.50, "target_bpm": 96,  "target_valence": 0.72, "target_danceability": 0.58, "target_acousticness": 0.70}

    # --- Adversarial / edge-case profiles ---

    # 1. BPM way out of the catalog range (BPM_MIN=55, BPM_MAX=168).
    #    Normalization produces a value > 1, so Gaussian diffs are inflated.
    #    Every song looks equally bad on tempo — does the scorer still rank sensibly?
    user_pref_extreme_bpm = {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.80, "target_bpm": 300,   # 300 BPM — physically impossible
        "target_valence": 0.80, "target_danceability": 0.80, "target_acousticness": 0.10,
    }

    # 2. Physically impossible combo: maximum energy AND maximum acousticness.
    #    Real songs with high energy are almost never acoustic. No song should score
    #    well on both simultaneously — who rises to the top?
    user_pref_contradictory = {
        "favorite_genre": "rock", "favorite_mood": "energetic",
        "target_energy": 1.0, "target_bpm": 160,
        "target_valence": 0.5, "target_danceability": 0.5, "target_acousticness": 1.0,
    }

    # 3. Nonexistent genre and mood — categorical bonuses can never fire.
    #    The recommender must rely purely on numerical similarity.
    #    Verifies the scorer doesn't crash or silently give wrong bonuses.
    user_pref_ghost_genre = {
        "favorite_genre": "space-jazz-metal", "favorite_mood": "melancholic-euphoria",
        "target_energy": 0.65, "target_bpm": 115,
        "target_valence": 0.65, "target_danceability": 0.65, "target_acousticness": 0.35,
    }

    # 4. All numerical targets pinned to 0.0 — the absolute floor.
    #    Songs with near-zero energy, BPM, valence, danceability, acousticness
    #    should score highest. Tests boundary behavior of Gaussian (diff close to 0.5+).
    user_pref_all_zeros = {
        "favorite_genre": "", "favorite_mood": "",
        "target_energy": 0.0, "target_bpm": 0,
        "target_valence": 0.0, "target_danceability": 0.0, "target_acousticness": 0.0,
    }

    # 5. All numerical targets pinned to 1.0 — the absolute ceiling.
    user_pref_all_ones = {
        "favorite_genre": "", "favorite_mood": "",
        "target_energy": 1.0, "target_bpm": 999,
        "target_valence": 1.0, "target_danceability": 1.0, "target_acousticness": 1.0,
    }

    # 6. Completely empty prefs dict — every .get() falls back to its default (0.5 / 120 / "").
    #    All songs should receive nearly identical scores; ranking becomes arbitrary.
    #    Good stress-test for division-by-zero or KeyError bugs.
    user_pref_empty = {}

    adversarial_profiles = [
        ("Extreme BPM (300)",            user_pref_extreme_bpm),
        ("Contradictory (max energy + max acousticness)", user_pref_contradictory),
        ("Ghost genre/mood",             user_pref_ghost_genre),
        ("All zeros",                    user_pref_all_zeros),
        ("All ones / max BPM",           user_pref_all_ones),
        ("Empty prefs (all defaults)",   user_pref_empty),
    ]

    recommendations = recommend_songs(user_pref, songs, k=5)

    print("\nUser preferences:")
    for key, value in user_pref.items():
        print(f"  {key}: {value}")

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()

    # --- Run adversarial profiles ---
    print("\n" + "=" * 60)
    print("ADVERSARIAL / EDGE-CASE PROFILE TESTS")
    print("=" * 60)

    for label, prefs in adversarial_profiles:
        print(f"\n[{label}]")
        print("  Prefs:", prefs)
        try:
            results = recommend_songs(prefs, songs, k=3)
            for song, score, explanation in results:
                print(f"  {song['title']} - Score: {score:.2f} | {explanation}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
