# 🎵 VibeCheck - Music Recommender

## Project Summary

In this project, a user provides input for a song they want to listen to. The application parses this query into a request to the Spotify API and returns 5 top ranked songs based on this query. Gemini provides a thoughtful reasoning for each song returned. This application allows users to find quick picks based on their mood, because sometimes searching through endless playlists is just too much.

---

## How The System Works

VibeCheck uses a four-step RAG (Retrieval-Augmented Generation) pipeline:

1. **Query parsing (Gemini)** — The user's natural language input (e.g. _"chill lo-fi for late night studying"_) is sent to Gemini, which converts it into a structured preference profile: target values for energy, valence, danceability, acousticness, and tempo, plus a genre and mood label.

2. **Track retrieval (Spotify)** — The parsed genre seeds are used to search the Spotify catalog for real candidate tracks. Audio features are requested from Spotify's `/audio-features` endpoint; if that is unavailable, per-track values are estimated by adding random variation around the user's targets so that each candidate is still distinct.

3. **Gaussian scoring (recommender)** — Every candidate track is scored against the user's preferences using a Gaussian similarity function: `score = e^(-(diff² / 2σ²))`, where `diff` is the absolute difference between the user's target and the song's value for each feature, and `σ = 0.2` is the tolerance. Features are weighted (energy 60%, valence 15%, danceability 12%, acousticness 8%, tempo 5%), with flat bonuses added for genre (+0.15) and mood (+0.10) matches. The top 5 songs are selected.

4. **Explanation generation (Gemini)** — The top 5 ranked songs are passed back to Gemini, which writes a short personalized explanation for each one grounded in the user's original query and the song's audio features.

---

## Architecture Overview

![System Design](<assets/diagrams/VibeCheck System Design.png>)

The design of the application is split into 5 essential levels:

- UI (Streamlit app)
- LLM (Google Gemini API, model `gemma-3-1b-it`)
- Music Catalog (Spotify Web API and local `data/songs.csv` as a fallback)
- Scoring (Pure Python using Gaussian similarity)
- Config (the local environment setup required to wire everything together)

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Create a .env in the root folder using .env.example as a guide

4. Sign up at developer.spotify.com, and create an App with the following settings:

   ```
   App Name: VibeCheck
   App description: Music recommender system.
   Redirect URIs: https://example.org/callback
   Which API/SDKs are you planning to use?: Web API
   ```

5. Add your Client ID and Client Secret to your .env file

6. Sign up for an API key at aistudio.google.com and add the key to your .env file

7. Run the app:

   ```bash
   streamlit run src/app.py
   ```

### Example inputs

- "chill lo-fi for late night studying"

  ```
  #1 Gravitation - Original Mix — Gennadiy Adamenko
  #2 Feel so Good - Florito
  #3 Love to Scat - Gary B
  #4 Nemo - Ingo Herrmann
  #5 Beat It - James Long
  ```

- "upbeat pop for a morning workout"

  ```
  #1 So Easy (To Fall in Love) - Olivia Dean
  #2 End of Beginning - Djo
  #3 American Girls - Harry Styles
  #4 E85 - Don Toliver
  #5 Stateside + Zara Larsson - PinkPantheress
  ```

- "something moody and acoustic for a rainy day"

  ```
  #1 Everybody Hurts - Thom Cooper
  #2 Grenade - Karizma Duo
  #3 New Heights - Denis Turbide
  #4 Raining - Ai Mougi
  # 5 Nothin Breaks Like a Heart - Landa
  ```

---

## Design Decisions

The system is built using simple scoring and an LLM's interpretation in two different areas of the application. This is done to allow retrieval from the Spotify API, and generation of the explanation for why the song was ranked in the top 5 based on the user's prompt. Only 5 songs are chosen as Spotify's API is particularly limiting with the number of songs it can return. The original plan was to have a "chatbot" like feel where the application can help a user deduce down a mood for a list of songs that fit that mood, but this was scrapped once I had found out just how little usage one can use with free tiers of the API keys. Instead, it was developed into a simple query processing system that parses a user's input into a mathematical representation, sent as a seed to Spotify's API and returns a list of top scoring songs based on the 1 LLM interpretation of the query. The results are returned with another AI generated explanation for why the song matched the query passed in.

---

## Limitations and Risks

The Spotify API is a major limitation as one can only return a few number of results. Another limitation is Gemini API. I had to use a lightweight model to avoid running into errors due to too many API requests being called. This may cause less than ideal matches and poor explanations. The limit on the API requests can also result in the application falling back on the data.csv, and output suggestions that don't exist or are just too far mismatched from what the user actually wants to listen to.

---

# TF Submission: Justin Dingeman

1. The core concept students needed to understand.

> Students should understand how to take a project and expand on its capabilities and functionality. They should be comfortable collaborating with and verifying AI implementations and output, as well as questioning it when they find something off about its suggestions.

2. Where students are most likely to struggle

> Students are likely to struggle with the initial set up since they may not understand using git that well, but the instructions are pretty straightforward. Elsewhere, they may struggle brining an idea to fruition or keeping control of the AI's implementations. It's very easy to get lost and let the AI take control of the project based on its understanding, rather than their own. If they can't properly express their ideas, they will not be able to push back on the AI properly to avoid it coming up with odd choices.

3. Where AI was helpful vs. misleading

> I used Claude Code. It was helpful in coming up with the functions and connecting the pieces together to get the app functioning. It was misleading in implementing the functions that call the Spotify API because it kept implementing deprecated functions, using incorrect limits, etc. It seemed like its knowledge was behind on the current state of the API.

4. One way they would guide a student without giving the answer

> For a large project like this, the risk is always getting too far into an AI implementation and not feeling comfortable to scale back on it. If they become lost, I would have them retrace their steps through the development and identify at which point they feel they lost the understanding of the implementation of the AI. Then, have them check out what exactly was done at that time and see if they can explain the implementation in their own words, even line by line.
