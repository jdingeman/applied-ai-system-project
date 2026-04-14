# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

VibeCheck

---

## 2. Intended Use

VibeCheck is a content-based music recommender designed to suggest songs that match a listener's stated taste preferences. Given a user profile — including a favorite genre, preferred mood, and target values for energy, valence, danceability, acousticness, and tempo — the system scores every song in the catalog and returns the top matches.

The recommender assumes that users can articulate their preferences as concrete numeric targets and categorical choices rather than through implicit listening history. It is intended for classroom exploration of recommender system concepts in an introductory AI course, not production use. There is no real user account system, no listening history, and no feedback loop — each recommendation is computed fresh from a hand-specified profile.

---

## 3. How the Model Works

Every song in the catalog is described by five numerical qualities — energy, mood positivity (valence), danceability, acousticness, and tempo — plus two categorical labels: genre and mood. A user profile supplies target values for each of those same qualities.

## To score a song, the system measures how close each of the song's numerical qualities is to what the user asked for. "Close" is not binary: a song that is slightly off still earns most of the credit, while a song that is far off earns very little. The five closeness scores are then blended into a single number using weights, so that energy counts for more than tempo in the final result. On top of that weighted score, the system adds a flat bonus if the song's genre matches the user's favorite genre, and a smaller bonus if the mood label matches. The final score is capped at 1.0.

## 4. Data

The intended data comes from Spotify's live catalog. The API gives access to certain audio features used to determine the best matched based on the LLM's interpretation of the numerical value that can be assigned to different keywords in the user's query. Fallback data in `data/songs.csv` is available in case the API call fails.

---

## 5. Strengths

It works relatively quickly and produces a short list in a matter of time that does appear thought out and well considered, instead of just appearing as a quick random selection of songs.

---

## 6. Limitations and Bias

Genre and mood bonus create a hard categorical ceilng so a user's favorite genre may dominate the top ranked songs. The Gaussian similarity scoring will penalize extreme values equally, so a user who prefers high energy would more likely tolerate energy that is too high rather than too low, but the similarity scoring penalizes them equally. BPM normalization is catalog-dependent, so it only depends on the bpm available in the songs.csv if it's used as a fallback.

---

## 7. Evaluation

It works well for the most part, but user input can be limiting. The numerical assignment of the keywords can cause the model to match to very specific values in Spotify's catalog, and will return songs they may not exactly match what the user is looking for. This is also due to the disparity between human experience and AI interpretation, and the language we use to express certain moods or ideas.

---

## 8. Future Work

I'd like to look more into the LLM's "decisions" on how it comes up with a user preference dict and how it assigns the values to look for based on the user's input. I think that pairing flexible language with pretty hard math rules can make for a less than ideal list of songs. If API limits were not of issue, I would want to come up with a much more complex system that does deep analysis of the user query and match it with songs - perhaps even by looking up how songs have been described by other users online.

---

## 9. Personal Reflection

I learned that recommender systems do some very complex math and that LLMs can prompt themselves to refine results. I also learned a bit more about streamlit and caching session tokens to prevent the entire session from being wiped each time the user types something.

---

## 10. Reliability and Evaluation

**Automated tests** — `tests/test_recommender.py` contains two pytest tests that verify the `Recommender` class: one confirms results are returned in ranked order with the best genre/mood match first, and one confirms that `explain_recommendation` always returns a non-empty string. A separate CLI tool, `src/scorer_cli.py`, runs 9 granular unit tests directly against `score_song()` (baseline perfect match, genre/mood bonus isolation, energy weight dominance, BPM boundary safety, and empty prefs) plus 6 adversarial edge-case profiles against the full catalog. All 9 granular tests pass; the adversarial profiles complete without exceptions.

**Confidence scoring** — every recommendation surfaces a numerical score in [0.0, 1.0] that is displayed to the user as a progress bar. A score near 1.0 means the song closely matched across all features; a lower score signals a weaker fit. This gives the user a built-in signal for how confident the system is in each result.

**Logging and error handling** — `src/logger.py` configures two rotating log files: `logs/app.log` captures the full pipeline trace at INFO level (parsed prefs, Spotify call results, scoring outcomes, AI explanation responses) and `logs/error.log` captures exceptions and fallback events at ERROR level. API failures — Spotify returning no results, Gemini failing to parse — are caught explicitly and fall back to the local catalog or the mechanical score explanation, so the app never surfaces a raw crash to the user.

**Human evaluation** — the app includes a "Show technical details" expander that lets the user inspect the structured preferences Gemini extracted from their query alongside the raw score for each result. This functions as a human checkpoint: if the parsed preferences look wrong, the user can rephrase and rerun. A developer debug panel also lets anyone manually trigger adversarial profiles against the live scorer to spot regressions.

## 11. Reflection and Ethics

Because energy is weighted so heavily, the system is more sensitive to that one quality than everything else combined. A song that perfectly matches your mood, genre, and tempo but has slightly the wrong energy level will still rank lower than a song that nails the energy. The genre and mood bonuses are also blunt — "pop" is treated as a single category, even though pop covers an enormous range of sounds.

The AI's translation of natural language into numbers is another weak point. Words like "moody" or "chill" don't map to a single value, and the model may interpret them differently each time. People with more niche or genre-fluid taste tend to get worse results because the system was designed around more standard musical vocabulary.

On the misuse side, the app only ever returns song recommendations, so there is not much harm it can do directly. The bigger long-term risk is a filter bubble: if someone always queries in the same way, the system will keep reinforcing the same taste without ever pushing them toward something new. There is also no way for the app to know if a user is gaming the query to get a specific result. The Spotify API credentials are the only genuinely sensitive part of the system, and those are kept out of the code in a local environment file that is never logged or displayed.

The AI's reliability is limited when you ask it to do complex, multi-level processes. The AI is going to interpret input, send it, and interpret results again based on it own interpretation. This is a highly limiting factor. Another surprising thing is that if you ask for something that it interprets as being of high energy, it'll almost always suggestion something by Powderfinger because of its closeness in energy level to the interpretation.

As mentioned, the AI used for collaboration is a bit behind on available APIs, so its initial suggestions for implementing the calls to Spotify's API were not correct due to those APIs either being deprecated, or because more stringent limitations were placed on those APIs.
