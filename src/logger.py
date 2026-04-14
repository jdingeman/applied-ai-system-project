"""
Logging configuration for the Music Recommender application.

Two log files are written to the project-level logs/ directory:
    logs/app.log   — INFO and above (full pipeline trace: queries, parsed prefs,
                     Spotify results, scoring, AI explanations)
    logs/error.log — ERROR and above only (exceptions, API failures, fallbacks
                     that required default values)

Both files rotate at 5 MB and keep 3 backups so disk usage stays bounded.

Usage — call setup_logging() once at app startup (app.py does this), then in
every other module:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Resolve logs/ relative to this file's location (src/ → project root → logs/)
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False  # guard against duplicate handler registration on Streamlit reruns


def setup_logging() -> None:
    """
    Configure application-wide logging. Safe to call multiple times — only
    the first call has any effect.

    Creates logs/ if it does not exist, then attaches two RotatingFileHandlers
    to the root logger:
        app.log   — INFO+
        error.log — ERROR+
    """
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(LOGS_DIR, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # app.log — INFO and above (full operational trace)
    app_handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, "app.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    # error.log — ERROR and above only
    error_handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, "error.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter; root must pass everything through
    root.addHandler(app_handler)
    root.addHandler(error_handler)
