"""
Command line runner for the Music Recommender.

Usage:
  python src/main.py                 # runs both sections
  python src/main.py --granular      # score_song unit tests only
  python src/main.py --adversarial   # edge-case profile tests only

Sections
--------
1. GRANULAR TESTS  --call score_song() directly on synthetic songs, one
   variable at a time. Add new entries to GRANULAR_TESTS to grow the suite.

2. ADVERSARIAL PROFILES --call recommend_songs() on edge-case preference
   dicts against the full CSV catalog.
"""

import sys
from typing import Callable, Dict, List, Optional

from recommender import load_songs, recommend_songs, score_song


# ─── Shared test fixtures ──────────────────────────────────────────────────
# A "perfect" synthetic song that exactly matches _BASE_PREFS.
# Override individual keys with {**_BASE_SONG, "energy": 0.30} in each test.

_BASE_SONG: Dict = {
    "id": 999, "title": "Synthetic Track", "artist": "Test",
    "genre": "pop", "mood": "happy",
    "energy": 0.80, "tempo_bpm": 120, "valence": 0.80,
    "danceability": 0.80, "acousticness": 0.15,
}

_BASE_PREFS: Dict = {
    "favorite_genre": "pop", "favorite_mood": "happy",
    "target_energy": 0.80, "target_bpm": 120,
    "target_valence": 0.80, "target_danceability": 0.80,
    "target_acousticness": 0.15,
}


# ─── Granular test registry ────────────────────────────────────────────────
# Each entry is a dict with:
#
#   name         str                        --what is being tested
#   prefs        Dict                       --passed to score_song as user_prefs
#   song         Dict                       --passed to score_song as song
#   expect       Callable[[float], bool]    --optional assertion on the score
#   expect_label str                        --human description of the assertion
#
# Tests without an "expect" key are printed as informational (marked "·").
# They are useful for side-by-side comparisons where the assertion would be
# expressed across two entries (e.g. "previous score < this score").

GRANULAR_TESTS: List[Dict] = [

    # ── Baseline ───────────────────────────────────────────────────────────
    {
        "name": "Perfect match -> score == 1.0",
        "prefs": _BASE_PREFS,
        "song": _BASE_SONG,
        "expect": lambda s: s == 1.0,
        "expect_label": "== 1.0",
    },

    # ── Genre / mood bonus isolation ───────────────────────────────────────
    # target_energy=0.50 creates a d0.30 energy gap, keeping the numerical
    # score ~0.59 so bonuses are visible and not clipped by the 1.0 ceiling.
    {
        "name": "Genre bonus (+0.15) fires --genre matches",
        "prefs": {**_BASE_PREFS, "target_energy": 0.50},
        "song": _BASE_SONG,
        "expect": lambda s: s > 0.75,
        "expect_label": "> 0.75 (numerical ~0.59 + genre 0.15 + mood 0.10)",
    },
    {
        "name": "Genre bonus absent --genre mismatches (jazz != pop)",
        "prefs": {**_BASE_PREFS, "favorite_genre": "jazz", "target_energy": 0.50},
        "song": _BASE_SONG,
        "expect": lambda s: s < 0.75,
        "expect_label": "< 0.75 (numerical ~0.59 + mood 0.10 only)",
    },
    {
        "name": "Mood bonus (+0.10) isolated --mood matches, genre mismatches",
        "prefs": {**_BASE_PREFS, "favorite_genre": "jazz", "target_energy": 0.50},
        "song": _BASE_SONG,
        "expect": lambda s: s > 0.65,
        "expect_label": "> 0.65 (numerical ~0.59 + mood 0.10)",
    },
    {
        "name": "No bonuses --both genre and mood mismatch",
        "prefs": {**_BASE_PREFS, "favorite_genre": "jazz", "favorite_mood": "moody",
                  "target_energy": 0.50},
        "song": _BASE_SONG,
        "expect": lambda s: s < 0.65,
        "expect_label": "< 0.65 (pure numerical ~0.59, no bonuses)",
    },

    # ── Energy weight dominance ─────────────────────────────────────────────
    # Same absolute diff (d0.20) on energy (weight 0.60) vs valence (weight 0.15).
    # Bonuses disabled so the comparison is pure numerical.
    # Expected: energy-off score (~0.76) < valence-off score (~0.94).
    {
        "name": "Energy off d0.20 --weight 0.60 --informational, compare with next",
        "prefs": {**_BASE_PREFS, "favorite_genre": "jazz", "favorite_mood": "moody"},
        "song": {**_BASE_SONG, "energy": 0.60},
    },
    {
        "name": "Valence off d0.20 --weight 0.15 --expect higher score than previous",
        "prefs": {**_BASE_PREFS, "favorite_genre": "jazz", "favorite_mood": "moody"},
        "song": {**_BASE_SONG, "valence": 0.60},
    },

    # ── BPM boundary safety ─────────────────────────────────────────────────
    {
        "name": "BPM below catalog min (55) --no crash, score in [0, 1]",
        "prefs": {**_BASE_PREFS, "target_bpm": 20},
        "song": {**_BASE_SONG, "tempo_bpm": 55},
        "expect": lambda s: 0.0 <= s <= 1.0,
        "expect_label": "in [0.0, 1.0]",
    },
    {
        "name": "BPM above catalog max (168) --no crash, score in [0, 1]",
        "prefs": {**_BASE_PREFS, "target_bpm": 999},
        "song": {**_BASE_SONG, "tempo_bpm": 168},
        "expect": lambda s: 0.0 <= s <= 1.0,
        "expect_label": "in [0.0, 1.0]",
    },

    # ── Missing / empty prefs ──────────────────────────────────────────────
    {
        "name": "Empty prefs dict --all .get() defaults fire, no crash",
        "prefs": {},
        "song": _BASE_SONG,
        "expect": lambda s: 0.0 <= s <= 1.0,
        "expect_label": "in [0.0, 1.0]",
    },
]


def run_granular_tests() -> None:
    print("\n" + "=" * 60)
    print("GRANULAR UNIT TESTS --score_song()")
    print("=" * 60)

    passed = failed = informational = 0

    for test in GRANULAR_TESTS:
        try:
            score, explanation = score_song(test["prefs"], test["song"])
        except Exception as exc:
            print(f"\n  [✗] {test['name']}")
            print(f"       EXCEPTION: {exc}")
            failed += 1
            continue

        expect: Optional[Callable[[float], bool]] = test.get("expect")

        if expect is None:
            tag = " . "
            informational += 1
        elif expect(score):
            tag = "PASS"
            passed += 1
        else:
            tag = "FAIL"
            failed += 1

        print(f"\n  [{tag}] {test['name']}")
        print(f"       score={score:.3f}  |  {explanation}")
        if expect is not None and not expect(score):
            print(f"       expected: {test.get('expect_label', '?')}")

    print(f"\n  {passed} passed | {failed} failed | {informational} informational")


# ─── Adversarial profiles ──────────────────────────────────────────────────

_ADVERSARIAL_PROFILES: List[tuple] = [
    # 1. BPM way out of catalog range --normalization produces value > 1.
    ("Extreme BPM (300)", {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.80, "target_bpm": 300,
        "target_valence": 0.80, "target_danceability": 0.80, "target_acousticness": 0.10,
    }),
    # 2. Impossible combo: max energy AND max acousticness simultaneously.
    ("Contradictory (max energy + max acousticness)", {
        "favorite_genre": "rock", "favorite_mood": "intense",
        "target_energy": 1.0, "target_bpm": 160,
        "target_valence": 0.5, "target_danceability": 0.5, "target_acousticness": 1.0,
    }),
    # 3. Genre and mood that don't exist --categorical bonuses can never fire.
    ("Ghost genre/mood", {
        "favorite_genre": "space-jazz-metal", "favorite_mood": "melancholic-euphoria",
        "target_energy": 0.65, "target_bpm": 115,
        "target_valence": 0.65, "target_danceability": 0.65, "target_acousticness": 0.35,
    }),
    # 4. All numerical targets at the floor.
    ("All zeros", {
        "favorite_genre": "", "favorite_mood": "",
        "target_energy": 0.0, "target_bpm": 0,
        "target_valence": 0.0, "target_danceability": 0.0, "target_acousticness": 0.0,
    }),
    # 5. All numerical targets at the ceiling.
    ("All ones / max BPM", {
        "favorite_genre": "", "favorite_mood": "",
        "target_energy": 1.0, "target_bpm": 999,
        "target_valence": 1.0, "target_danceability": 1.0, "target_acousticness": 1.0,
    }),
    # 6. Completely empty --every .get() falls back to its default.
    ("Empty prefs (all defaults)", {}),
]


def run_adversarial_tests(songs: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("ADVERSARIAL / EDGE-CASE PROFILE TESTS --recommend_songs()")
    print("=" * 60)

    for label, prefs in _ADVERSARIAL_PROFILES:
        print(f"\n[{label}]")
        print("  Prefs:", prefs)
        try:
            results = recommend_songs(prefs, songs, k=3)
            for song, score, explanation in results:
                print(f"  {song['title']} --Score: {score:.2f} | {explanation}")
        except Exception as exc:
            print(f"  ERROR: {exc}")


# ─── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from catalog.")

    args = sys.argv[1:]
    run_all = not args or "--all" in args

    if run_all or "--granular" in args:
        run_granular_tests()

    if run_all or "--adversarial" in args:
        run_adversarial_tests(songs)


if __name__ == "__main__":
    main()
