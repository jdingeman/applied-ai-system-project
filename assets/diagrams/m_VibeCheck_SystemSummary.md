flowchart TD
A[/"👤 User\nNatural language query"/]
B["Gemini — Query Parser\nparse_natural_language_query"]
C["Spotify API\nfetch_spotify_recommendations"]
D["Recommender\nscore_song · recommend_songs"]
E["Gemini — RAG Explainer\ngenerate_rag_explanation"]
F[/"👤 User\nSong cards + AI explanations"/]

    CSV[("data/songs.csv\nFallback")]

    A -->|"raw text"| B
    B -->|"structured prefs dict"| C
    C -->|"live tracks\n(jittered features if 403)"| D
    C -->|"Spotify fails"| CSV
    CSV -->|"20 local songs"| D
    D -->|"top-10 ranked songs"| E
    E -->|"5 enriched results"| F
