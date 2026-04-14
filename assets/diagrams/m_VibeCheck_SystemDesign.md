flowchart TD
subgraph EXT["☁️ External Services"]
SPOTIFY["🎵 Spotify Web API\n/search + /audio-features\n(/recommendations deprecated Nov 2023)"]
GEMINI["🤖 Google Gemini API\ngemma-3-1b-it"]
end

    subgraph APP["src/app.py — Streamlit UI"]
        INPUT[/"👤 User Input\nNatural language query\ne.g. 'chill lo-fi for studying'"/]
        RESULTS[/"📋 Results Display\nSong cards · Score bars\nAI explanations · Spotify links\n[Expander: parsed prefs JSON]"/]
    end

    subgraph PIPELINE["⚙️ Core Pipeline"]
        QP["src/query_parser.py\nparse_natural_language_query\n──────────────────\nNL → structured prefs dict\n{seed_genres, target_energy,\ntarget_bpm, favorite_genre ...}"]
        SC["src/spotify_client.py\nfetch_spotify_recommendations\n──────────────────\nLive tracks via /search by genre\nJittered per-track fallback features\nwhen /audio-features returns 403"]
        REC["src/recommender.py  ✅ UNCHANGED\nrecommend_songs + score_song\n──────────────────\nGaussian similarity · Weights\nCategorical bonuses\nReturns top-10 ranked tuples"]
        RAG["src/rag_explainer.py\ngenerate_rag_explanation\n──────────────────\nEnriches top-5 with\nnatural language ai_explanation"]
    end

    subgraph FALLBACK["🛡️ Reliability / Fallback"]
        CSV[("data/songs.csv\nLocal catalog\n20 songs")]
        MECHEXP["mechanical_explanation\nfrom score_song"]
    end

    subgraph CHECKS["👤 Human & 🧪 Testing Checkpoints"]
        H1["👤 Human Checkpoint 1\nUser reads & refines\ntheir own query before submit"]
        H2["👤 Human Checkpoint 2\nUser reviews results · inspects\nparsed prefs JSON in expander\nCan re-query if unsatisfied"]
        T1["🧪 Unit Tests\ntests/test_recommender.py\nScoring + explanation tests\nRuns fully offline"]
        T2["🧪 Adversarial Profiles\nsrc/scorer_cli.py\n6 edge-case stress tests\nextreme BPM, empty prefs, etc."]
    end

    INPUT -->|"raw query string"| QP
    QP <-->|"Gemini API call #1\nJSON output"| GEMINI
    QP -->|"structured_prefs dict"| SC
    SC <-->|"Spotify API calls\ntracks + audio features"| SPOTIFY
    SC -->|"List[Dict] — candidate tracks"| REC
    REC -->|"top-10 (song, score, explanation)"| RAG
    RAG <-->|"Gemini API call #2\nJSON array of explanations"| GEMINI
    RAG -->|"5 enriched results"| RESULTS

    SC -->|"Spotify unavailable\nst.warning shown"| CSV
    CSV -->|"fallback tracks"| REC
    RAG -->|"Gemini call fails"| MECHEXP
    MECHEXP -->|"fallback explanation"| RESULTS

    H1 -. "validates input" .-> INPUT
    H2 -. "inspects output" .-> RESULTS
    T1 -. "tests scoring logic" .-> REC
    T2 -. "stress-tests edge cases" .-> REC
