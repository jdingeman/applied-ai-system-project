import json
import logging
import os
import re
from typing import Dict

from google import genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemma-3-1b-it"

load_dotenv()

# Spotify only accepts seed genres from this approved list.
# Gemini is constrained to these values so Spotify never rejects the request.
VALID_SPOTIFY_GENRES = [
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
    "black-metal", "bluegrass", "blues", "bossanova", "brazil", "breakbeat",
    "british", "cantopop", "chicago-house", "children", "chill", "classical",
    "club", "comedy", "country", "dance", "dancehall", "death-metal",
    "deep-house", "detroit-techno", "disco", "disney", "drum-and-bass", "dub",
    "dubstep", "edm", "electro", "electronic", "emo", "folk", "forro",
    "french", "funk", "garage", "german", "gospel", "goth", "grindcore",
    "groove", "grunge", "guitar", "happy", "hard-rock", "hardcore", "hardstyle",
    "heavy-metal", "hip-hop", "holidays", "honky-tonk", "house", "idm",
    "indian", "indie", "indie-pop", "industrial", "iranian", "j-dance",
    "j-idol", "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin",
    "latino", "malay", "mandopop", "metal", "metal-misc", "metalcore",
    "minimal-techno", "movies", "mpb", "new-age", "new-release", "opera",
    "pagode", "party", "philippines-opm", "piano", "pop", "pop-film",
    "post-dubstep", "power-pop", "progressive-house", "psych-rock", "punk",
    "punk-rock", "r-n-b", "rainy-day", "reggae", "reggaeton", "road-trip",
    "rock", "rock-n-roll", "rockabilly", "romance", "sad", "salsa", "samba",
    "sertanejo", "show-tunes", "singer-songwriter", "ska", "sleep", "songwriter",
    "soul", "soundtracks", "spanish", "study", "summer", "swedish", "synth-pop",
    "tango", "techno", "trance", "trip-hop", "turkish", "work-out", "world-music",
]

# Moods must match the four labels derive_mood() in spotify_client.py produces,
# so the mood bonus in score_song() can fire when user and track moods align.
VALID_MOODS = ["happy", "intense", "chill", "moody"]

# Safe neutral defaults used when Gemini's output cannot be parsed after retrying.
_DEFAULTS: Dict = {
    "seed_genres":          ["pop"],
    "target_energy":        0.5,
    "target_valence":       0.5,
    "target_danceability":  0.5,
    "target_acousticness":  0.5,
    "target_bpm":           120.0,
    "favorite_genre":       "pop",
    "favorite_mood":        "chill",
    "query_summary":        "General music recommendation",
}


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add around JSON output."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def parse_natural_language_query(
    user_query: str,
    model: genai.Client,
) -> Dict:
    """
    Send the user's natural language query to Gemini and return a structured
    dict of audio preferences compatible with both score_song() and
    fetch_spotify_recommendations().

    Gemini API call #1 in the RAG pipeline.

    Returns a dict with keys:
        seed_genres         List[str]  — 1-3 valid Spotify genre seeds
        target_energy       float      — 0.0 (silent/calm) to 1.0 (intense)
        target_valence      float      — 0.0 (negative/sad) to 1.0 (positive/happy)
        target_danceability float      — 0.0 (not danceable) to 1.0 (very danceable)
        target_acousticness float      — 0.0 (electronic) to 1.0 (fully acoustic)
        target_bpm          float      — beats per minute, typically 60–180
        favorite_genre      str        — primary genre string for score_song()
        favorite_mood       str        — one of: happy, intense, chill, moody
        query_summary       str        — Gemini's one-sentence paraphrase of the query

    On JSONDecodeError, retries once with a stricter prompt.
    Falls back to neutral _DEFAULTS if both attempts fail.
    """
    logger.info("Parsing query | query=%r", user_query)
    prompt = _build_parser_prompt(user_query)
    logger.info("Prompt built | prompt_length=%d chars | model=%s", len(prompt), _GEMINI_MODEL)

    try:
        response = model.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
        raw = _strip_markdown_fences(response.text or "")
        logger.info("Gemini response received | raw_length=%d chars", len(raw))
        logger.info("Raw Gemini response | text=%r", raw[:500])  # cap at 500 chars
        result = _validate_and_normalize(json.loads(raw))
        logger.info(
            "Parse successful | genres=%s | mood=%s | energy=%.2f | "
            "valence=%.2f | bpm=%.0f | summary=%r",
            result.get("seed_genres"),
            result.get("favorite_mood"),
            result.get("target_energy", 0.0),
            result.get("target_valence", 0.0),
            result.get("target_bpm", 0.0),
            result.get("query_summary", ""),
        )
        return result

    except json.JSONDecodeError as json_err:
        logger.warning(
            "JSON parse failed on first attempt, retrying with strict prompt | error=%s", json_err
        )
        # Retry once with an even stricter prompt
        try:
            strict_prompt = (
                "You must respond with ONLY a JSON object. "
                "No markdown, no code fences, no explanation. "
                "Just the raw JSON.\n\n" + prompt
            )
            response = model.models.generate_content(model=_GEMINI_MODEL, contents=strict_prompt)
            raw = _strip_markdown_fences(response.text or "")
            logger.info("Strict-prompt response received | raw_length=%d chars", len(raw))
            result = _validate_and_normalize(json.loads(raw))
            logger.info("Parse successful on retry | genres=%s | mood=%s", result.get("seed_genres"), result.get("favorite_mood"))
            return result

        except Exception as e:
            logger.error(
                "Both parse attempts failed, using defaults | error=%s",
                e,
                exc_info=True,
            )
            return dict(_DEFAULTS)

    except Exception as e:
        logger.error("Gemini API error, using defaults | error=%s", e, exc_info=True)
        return dict(_DEFAULTS)


def _build_parser_prompt(user_query: str) -> str:
    """
    Build the prompt that instructs Gemini to convert the user's natural
    language query into a structured JSON object.

    The prompt:
    - Constrains seed_genres to VALID_SPOTIFY_GENRES so Spotify never rejects them
    - Provides audio feature scale anchors so Gemini maps words to numbers accurately
    - Constrains favorite_mood to the four labels derive_mood() produces
    - Demands raw JSON output with no surrounding prose
    """
    genre_list = ", ".join(VALID_SPOTIFY_GENRES)
    mood_list = ", ".join(VALID_MOODS)

    return f"""You are a music preference interpreter. Convert the user's query into a JSON object describing their ideal music.

USER QUERY: "{user_query}"

OUTPUT FORMAT — respond with ONLY this JSON object, no markdown fences, no explanation:
{{
  "seed_genres": ["<genre1>", "<genre2>"],
  "target_energy": <0.0-1.0>,
  "target_valence": <0.0-1.0>,
  "target_danceability": <0.0-1.0>,
  "target_acousticness": <0.0-1.0>,
  "target_bpm": <60-200>,
  "favorite_genre": "<single genre string>",
  "favorite_mood": "<mood>",
  "query_summary": "<one sentence paraphrase of the user's intent>"
}}

RULES:
1. seed_genres must contain 1–3 values chosen ONLY from this list:
   {genre_list}

2. Audio feature scale anchors:
   - energy:        0.1 = whisper-quiet acoustic ballad, 0.5 = mid-tempo pop, 0.9 = aggressive EDM or metal
   - valence:       0.1 = deeply sad or dark, 0.5 = neutral, 0.9 = euphoric or joyful
   - danceability:  0.1 = free-form jazz or classical, 0.5 = light groove, 0.9 = club track built for dancing
   - acousticness:  0.1 = fully electronic/produced, 0.5 = mixed, 0.9 = unplugged/acoustic instruments only
   - target_bpm:    60 = very slow ballad, 90 = relaxed groove, 120 = pop tempo, 150 = fast dance, 180 = drum and bass

3. favorite_mood must be exactly one of: {mood_list}
   - happy   = upbeat, positive, energetic (high valence + high energy)
   - intense = driving, aggressive, tense (low valence + high energy)
   - chill   = relaxed, easygoing, light  (high valence + low energy)
   - moody   = melancholic, dark, reflective (low valence + low energy)

4. favorite_genre must be a single string from the seed_genres list.

5. query_summary should reflect the user's words faithfully in one sentence."""


def _validate_and_normalize(raw: Dict) -> Dict:
    """
    Validate Gemini's parsed JSON and fill in safe defaults for any
    missing, wrong-type, or out-of-range values.
    """
    result = dict(_DEFAULTS)

    # seed_genres — filter to only valid Spotify genres
    genres = raw.get("seed_genres", [])
    if isinstance(genres, list):
        valid = [g for g in genres if g in VALID_SPOTIFY_GENRES]
        if valid:
            result["seed_genres"] = valid[:3]

    # Numeric features — clamp to valid ranges
    for key, lo, hi in [
        ("target_energy",        0.0, 1.0),
        ("target_valence",       0.0, 1.0),
        ("target_danceability",  0.0, 1.0),
        ("target_acousticness",  0.0, 1.0),
    ]:
        val = raw.get(key)
        if isinstance(val, (int, float)):
            result[key] = max(lo, min(hi, float(val)))

    bpm = raw.get("target_bpm")
    if isinstance(bpm, (int, float)):
        result["target_bpm"] = max(40.0, min(220.0, float(bpm)))

    # favorite_genre — fall back to first seed genre
    fg = raw.get("favorite_genre", "")
    if isinstance(fg, str) and fg in VALID_SPOTIFY_GENRES:
        result["favorite_genre"] = fg
    elif result["seed_genres"]:
        result["favorite_genre"] = result["seed_genres"][0]

    # favorite_mood — must be one of the four derive_mood() labels
    mood = raw.get("favorite_mood", "")
    if isinstance(mood, str) and mood in VALID_MOODS:
        result["favorite_mood"] = mood

    # query_summary — plain string, used by rag_explainer
    summary = raw.get("query_summary", "")
    if isinstance(summary, str) and summary.strip():
        result["query_summary"] = summary.strip()

    return result
