import json
import logging
import re
from typing import Dict, List, Tuple

from google import genai

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemma-3-1b-it"


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add around JSON output."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def generate_rag_explanation(
    original_query: str,
    query_summary: str,
    top_songs: List[Tuple[Dict, float, str]],
    model: genai.Client,
) -> List[Dict]:
    """
    Gemini API call #2 — the Generation step of the RAG pipeline.

    Takes the top-ranked songs from recommend_songs() and generates a
    personalized natural language explanation for each one, grounded in
    the user's original query.

    A single Gemini call is made for all songs to minimize latency and
    token cost. If the call fails, each song falls back to its mechanical
    explanation from score_song() so the UI always has something to show.

    Args:
        original_query:  The raw text the user typed.
        query_summary:   Gemini's one-sentence paraphrase from query_parser.
        top_songs:       List of (song_dict, score, mechanical_explanation)
                         tuples from recommend_songs(), already sorted best-first.
        model:           Configured Gemini GenerativeModel instance.

    Returns:
        List of enriched dicts, one per song, with keys:
            id, title, artist, genre, mood,
            energy, tempo_bpm, valence, danceability, acousticness,
            spotify_url, score, mechanical_explanation, ai_explanation
    """
    logger.info(
        "Generating explanations | query=%r | song_count=%d | model=%s",
        original_query, len(top_songs), _GEMINI_MODEL,
    )
    enriched = _build_enriched(top_songs)

    try:
        prompt = _build_explainer_prompt(original_query, query_summary, top_songs)
        logger.info("Explainer prompt built | prompt_length=%d chars", len(prompt))

        response = model.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
        raw = _strip_markdown_fences(response.text or "")
        logger.info("Gemini response received | raw_length=%d chars", len(raw))
        logger.info("Raw Gemini response | text=%r", raw[:500])  # cap at 500 chars

        explanations = json.loads(raw)

        if not isinstance(explanations, list):
            raise ValueError(f"Gemini returned non-list JSON: {type(explanations).__name__}")

        # Match explanations back to songs by position.
        # If Gemini returns fewer items than songs, the zip stops early
        # and remaining songs keep their mechanical_explanation.
        matched = 0
        for song_dict, explanation_item in zip(enriched, explanations):
            ai_text = explanation_item.get("ai_explanation", "")
            if isinstance(ai_text, str) and ai_text.strip():
                song_dict["ai_explanation"] = ai_text.strip()
                matched += 1

        logger.info("Explanations extracted | matched=%d / requested=%d", matched, len(top_songs))

    except Exception as e:
        logger.error(
            "Gemini explanation failed, using mechanical fallbacks | error=%s",
            e,
            exc_info=True,
        )

    return enriched


def _build_enriched(top_songs: List[Tuple[Dict, float, str]]) -> List[Dict]:
    """
    Flatten (song_dict, score, mechanical_explanation) tuples into dicts
    and pre-populate ai_explanation with the mechanical fallback.
    """
    result = []
    for song, score, mechanical in top_songs:
        entry = dict(song)
        entry["score"] = round(score, 3)
        entry["mechanical_explanation"] = mechanical
        # Default: mechanical explanation. Overwritten if Gemini succeeds.
        entry["ai_explanation"] = mechanical
        result.append(entry)
    return result


def _build_explainer_prompt(
    original_query: str,
    query_summary: str,
    top_songs: List[Tuple[Dict, float, str]],
) -> str:
    """
    Build the prompt that asks Gemini to write a personalized explanation
    for each recommended song.

    All songs are passed in a single prompt to avoid multiple API calls.
    Gemini is instructed to return a JSON array — one object per song —
    so explanations can be matched back by position.
    """
    song_lines = []
    for i, (song, score, mechanical) in enumerate(top_songs, start=1):
        song_lines.append(
            f"{i}. \"{song['title']}\" by {song['artist']} "
            f"(score: {score:.2f}, energy: {song['energy']:.2f}, "
            f"valence: {song['valence']:.2f}, tempo: {song['tempo_bpm']:.0f} BPM, "
            f"acousticness: {song['acousticness']:.2f}, "
            f"danceability: {song['danceability']:.2f}) "
            f"— scoring reason: {mechanical}"
        )

    songs_block = "\n".join(song_lines)

    return f"""You are a music recommendation assistant. A user searched for music with this request:

USER QUERY: "{original_query}"
SUMMARY: {query_summary}

The following songs were retrieved from Spotify and ranked by audio feature similarity to the user's request:

{songs_block}

Write a short, personalized explanation (1–2 sentences) for why each song suits the user's request.
Ground each explanation in the specific audio features and the user's stated intent.
Use second-person ("This track...", "Its ..."), avoid generic phrases like "perfect for you".

Respond with ONLY a JSON array — no markdown, no prose outside the JSON:
[
  {{"ai_explanation": "<explanation for song 1>"}},
  {{"ai_explanation": "<explanation for song 2>"}},
  ...
]"""
