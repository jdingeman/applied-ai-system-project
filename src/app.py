import logging
import os
import sys
from typing import Dict, List

from google import genai
import streamlit as st
from dotenv import load_dotenv

# Allow sibling imports (query_parser, spotify_client, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Logging must be configured before any sibling module is imported so that
# every module-level getLogger(__name__) call picks up our handlers.
from logger import setup_logging
setup_logging()

from query_parser import parse_natural_language_query
from rag_explainer import generate_rag_explanation
from recommender import load_songs, recommend_songs
from spotify_client import fetch_spotify_recommendations, get_spotify_client

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adversarial profiles — mirrored from src/scorer_cli.py for in-app debug testing
# ---------------------------------------------------------------------------

ADVERSARIAL_PROFILES: Dict[str, Dict] = {
    "Extreme BPM (300)": {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.80, "target_bpm": 300,
        "target_valence": 0.80, "target_danceability": 0.80, "target_acousticness": 0.10,
    },
    "Contradictory (max energy + max acousticness)": {
        "favorite_genre": "rock", "favorite_mood": "intense",
        "target_energy": 1.0, "target_bpm": 160,
        "target_valence": 0.5, "target_danceability": 0.5, "target_acousticness": 1.0,
    },
    "Ghost genre / mood": {
        "favorite_genre": "space-jazz-metal", "favorite_mood": "melancholic-euphoria",
        "target_energy": 0.65, "target_bpm": 115,
        "target_valence": 0.65, "target_danceability": 0.65, "target_acousticness": 0.35,
    },
    "All zeros": {
        "favorite_genre": "", "favorite_mood": "",
        "target_energy": 0.0, "target_bpm": 0,
        "target_valence": 0.0, "target_danceability": 0.0, "target_acousticness": 0.0,
    },
    "All ones / max BPM": {
        "favorite_genre": "", "favorite_mood": "",
        "target_energy": 1.0, "target_bpm": 999,
        "target_valence": 1.0, "target_danceability": 1.0, "target_acousticness": 1.0,
    },
    "Empty prefs (all defaults)": {},
}

# ---------------------------------------------------------------------------
# Cached resource initialisation — runs once per session, not on every render
# ---------------------------------------------------------------------------

@st.cache_resource
def init_gemini_model() -> genai.Client:
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def init_spotify_client():
    return get_spotify_client()


@st.cache_resource
def load_fallback_songs() -> List[Dict]:
    """Load the local CSV catalog used when Spotify is unavailable."""
    songs_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    return load_songs(songs_path)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_rag_pipeline(user_query: str) -> List[Dict]:
    """
    Orchestrate the full RAG pipeline for a given user query.

    Steps:
      1. Claude parses the natural language query → structured prefs dict
      2. Spotify fetches 30 live candidate tracks (falls back to CSV on failure)
      3. recommend_songs() re-ranks candidates using Gaussian similarity scoring
      4. Claude generates a personalized explanation for each top-5 song

    Returns a list of up to 5 enriched song dicts ready for display.
    """
    logger.info("Pipeline started | query=%r | query_length=%d", user_query, len(user_query))

    gemini_model = init_gemini_model()
    spotify_client = init_spotify_client()
    fallback_songs = load_fallback_songs()

    try:
        with st.status("Finding your music...", expanded=True) as status:

            # Step 1 — Parse the natural language query
            st.write("Interpreting your request...")
            logger.info("Step 1 — Parsing query with Gemini")
            structured_prefs = parse_natural_language_query(user_query, model=gemini_model)
            st.session_state["structured_prefs"] = structured_prefs
            logger.info(
                "Step 1 — Query parsed | genres=%s | mood=%s | energy=%.2f | "
                "valence=%.2f | danceability=%.2f | acousticness=%.2f | bpm=%.0f | "
                "summary=%r",
                structured_prefs.get("seed_genres"),
                structured_prefs.get("favorite_mood"),
                structured_prefs.get("target_energy", 0.0),
                structured_prefs.get("target_valence", 0.0),
                structured_prefs.get("target_danceability", 0.0),
                structured_prefs.get("target_acousticness", 0.0),
                structured_prefs.get("target_bpm", 0.0),
                structured_prefs.get("query_summary", ""),
            )

            # Step 2 — Retrieve live tracks from Spotify
            st.write("Fetching tracks from Spotify...")
            logger.info(
                "Step 2 — Fetching Spotify tracks | seed_genres=%s",
                structured_prefs.get("seed_genres"),
            )
            candidate_tracks = []

            if spotify_client is not None:
                candidate_tracks = fetch_spotify_recommendations(
                    sp=spotify_client,
                    seed_genres=structured_prefs["seed_genres"],
                    target_energy=structured_prefs["target_energy"],
                    target_valence=structured_prefs["target_valence"],
                    target_danceability=structured_prefs["target_danceability"],
                    target_acousticness=structured_prefs["target_acousticness"],
                    target_tempo=structured_prefs["target_bpm"],
                )
                logger.info("Step 2 — Spotify returned %d candidate tracks", len(candidate_tracks))

            if not candidate_tracks:
                logger.warning(
                    "Step 2 — Spotify unavailable or returned no results; "
                    "falling back to local catalog | fallback_size=%d",
                    len(fallback_songs),
                )
                st.warning(
                    "Spotify is unavailable or returned no results — "
                    "using local catalog as fallback."
                )
                candidate_tracks = fallback_songs

            # Step 3 — Score and rank with existing Gaussian scorer
            st.write("Scoring and ranking...")
            logger.info("Step 3 — Scoring %d candidates", len(candidate_tracks))
            top_songs = recommend_songs(structured_prefs, candidate_tracks, k=10)
            logger.info(
                "Step 3 — Scoring complete | top_song=%r by %r | top_score=%.3f",
                top_songs[0][0].get("title") if top_songs else None,
                top_songs[0][0].get("artist") if top_songs else None,
                top_songs[0][1] if top_songs else 0.0,
            )

            # Step 4 — Generate AI explanations for top 5
            st.write("Generating personalised explanations...")
            logger.info("Step 4 — Requesting AI explanations for top %d songs", min(5, len(top_songs)))
            enriched = generate_rag_explanation(
                original_query=user_query,
                query_summary=structured_prefs.get("query_summary", user_query),
                top_songs=top_songs[:5],
                model=gemini_model,
            )

            status.update(label="Done!", state="complete", expanded=False)

        logger.info(
            "Pipeline complete | query=%r | results_count=%d | titles=%s",
            user_query,
            len(enriched),
            [s.get("title") for s in enriched],
        )
        return enriched

    except Exception:
        logger.error(
            "Pipeline failed | query=%r",
            user_query,
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def render_song_card(song: Dict, rank: int) -> None:
    """Render a single recommendation card inside a styled container."""
    with st.container(border=True):
        col_rank, col_content = st.columns([1, 9])

        with col_rank:
            st.markdown(f"### #{rank}")

        with col_content:
            st.markdown(f"**{song.get('title', 'Unknown')}** — {song.get('artist', 'Unknown')}")

            tag_col, score_col = st.columns([3, 2])
            with tag_col:
                genre = song.get("genre", "—")
                mood = song.get("mood", "—")
                st.caption(f"Genre: `{genre}`   Mood: `{mood}`")
            with score_col:
                score = song.get("score", 0.0)
                st.progress(score, text=f"Score: {score:.2f}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Energy",       f"{song.get('energy', 0):.2f}")
            m2.metric("Valence",      f"{song.get('valence', 0):.2f}")
            m3.metric("Danceability", f"{song.get('danceability', 0):.2f}")
            m4.metric("BPM",          f"{song.get('tempo_bpm', 0):.0f}")

            st.markdown(f"> {song.get('ai_explanation', '')}")

            spotify_url = song.get("spotify_url", "")
            if spotify_url:
                st.link_button("Open on Spotify", spotify_url)


def render_technical_details(results: List[Dict], structured_prefs: Dict) -> None:
    """Collapsible section showing parsed prefs and raw scores — human checkpoint."""
    with st.expander("Show technical details"):
        st.subheader("Parsed preferences (from your query)")
        st.json(structured_prefs)

        st.subheader("Raw scores")
        import pandas as pd
        rows = [
            {
                "Title":        s.get("title"),
                "Artist":       s.get("artist"),
                "Score":        s.get("score"),
                "Energy":       s.get("energy"),
                "Valence":      s.get("valence"),
                "Danceability": s.get("danceability"),
                "BPM":          s.get("tempo_bpm"),
                "Acousticness": s.get("acousticness"),
            }
            for s in results
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Debug panel
# ---------------------------------------------------------------------------

def render_debug_panel() -> None:
    """
    Developer tool: run adversarial profiles directly against the local catalog,
    bypassing Gemini and Spotify entirely. Results use the mechanical explanation
    from score_song() — no LLM call is made.
    """
    st.divider()
    st.subheader("Debug: Adversarial Profile Tester")
    st.caption(
        "Runs `recommend_songs()` directly against the local CSV catalog. "
        "No Gemini or Spotify calls are made."
    )

    profile_name = st.selectbox("Profile", list(ADVERSARIAL_PROFILES.keys()))
    prefs = ADVERSARIAL_PROFILES[profile_name]

    st.json(prefs if prefs else {"note": "empty dict — all scorer .get() calls fall back to defaults"})

    if st.button("Run Test", type="secondary"):
        songs = load_fallback_songs()
        logger.info("Debug — Running adversarial profile | profile=%r | prefs=%s", profile_name, prefs)
        try:
            scored = recommend_songs(prefs, songs, k=5)
            logger.info(
                "Debug — Profile test complete | profile=%r | top_song=%r | top_score=%.3f",
                profile_name,
                scored[0][0].get("title") if scored else None,
                scored[0][1] if scored else 0.0,
            )
            st.session_state["debug_results"] = (profile_name, prefs, scored)
        except Exception as e:
            logger.error("Debug — Scorer raised an exception | profile=%r | error=%s", profile_name, e, exc_info=True)
            st.error(f"scorer raised an exception: {e}")
            st.session_state.pop("debug_results", None)

    if "debug_results" in st.session_state:
        label, ran_prefs, scored = st.session_state["debug_results"]
        if ran_prefs is not prefs:
            # Profile changed since last run — stale results, don't show
            pass
        else:
            st.markdown(f"**Results for:** `{label}`")
            for i, (song, score, explanation) in enumerate(scored, start=1):
                entry = dict(song)
                entry["score"] = round(score, 3)
                entry["ai_explanation"] = explanation
                render_song_card(entry, rank=i)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(spotify_available: bool) -> bool:
    with st.sidebar:
        st.title("About")
        st.markdown(
            "This app uses a **RAG pipeline** to recommend music:\n\n"
            "1. **Gemini** interprets your query\n"
            "2. **Spotify** retrieves live candidate tracks\n"
            "3. **Gaussian scoring** re-ranks by audio features\n"
            "4. **Gemini** explains why each track fits"
        )
        st.divider()
        st.subheader("Data source")
        if spotify_available:
            st.success("Live Spotify catalog")
        else:
            st.warning("Local fallback catalog (20 songs)")

        st.divider()
        debug_mode = st.checkbox("Developer / Debug Mode")
        st.caption("AI110 Applied AI System Project")

    return debug_mode


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Music Recommender",
        page_icon="🎵",
        layout="wide",
    )

    spotify_client = init_spotify_client()
    logger.info(
        "App started | spotify_available=%s",
        spotify_client is not None,
    )
    debug_mode = render_sidebar(spotify_available=spotify_client is not None)

    st.title("Music Recommender")
    st.markdown("Describe what you want to listen to and get personalised recommendations from Spotify.")

    st.caption(
        "Try: *\"chill lo-fi for late night studying\"* · "
        "*\"upbeat pop for a morning workout\"* · "
        "*\"something moody and acoustic for a rainy day\"*"
    )

    user_query = st.text_input(
        label="What are you in the mood for?",
        placeholder="e.g. relaxing jazz for a quiet evening",
        max_chars=200,
    )

    search_clicked = st.button("Find My Songs", type="primary")

    if search_clicked:
        if not user_query or len(user_query.strip()) < 3:
            st.warning("Please describe your mood or activity in a few words.")
            return

        results = run_rag_pipeline(user_query.strip())
        st.session_state["results"] = results
        st.session_state["last_query"] = user_query.strip()

    # Display results if they exist in session state
    if "results" in st.session_state:
        results = st.session_state["results"]
        last_query = st.session_state.get("last_query", "")

        st.divider()
        st.subheader(f"Top picks for: *\"{last_query}\"*")

        for i, song in enumerate(results, start=1):
            render_song_card(song, rank=i)

        # Technical details expander — human checkpoint #2
        structured_prefs = st.session_state.get("structured_prefs", {})
        render_technical_details(results, structured_prefs)

    if debug_mode:
        render_debug_panel()


if __name__ == "__main__":
    main()
