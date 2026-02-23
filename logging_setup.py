import logging
import os
from logging.handlers import RotatingFileHandler
from .context_request_id import get_request_id

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = get_request_id()
        except Exception:
            record.request_id = None
        return True

def setup_logging(level: str = None, log_file: str | None = "agent-debug.log"):
    """
    level: overall root log level (e.g. "INFO"). If None, read from env LOG_LEVEL.
    log_file: path to file (keep None to disable file logging).
    Console handler level controlled by LOG_CONSOLE_LEVEL (default WARNING) to avoid spamming interactive terminal.
    """
    root = logging.getLogger()
    # determine levels
    env_root_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root_level = getattr(logging, env_root_level, logging.INFO)
    console_level_name = os.getenv("LOG_CONSOLE_LEVEL", "WARNING").upper()
    console_level = getattr(logging, console_level_name, logging.WARNING)

    # remove old handlers
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(root_level)

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s [req:%(request_id)s]"
    formatter = logging.Formatter(fmt)

    # Console handler: set to console_level (default WARNING) to keep terminal clean
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler: keep more detailed logs if requested
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        # file logs use root_level so you can capture INFO/DEBUG into file while console stays quiet
        fh.setLevel(root_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # inject request_id field
    root.addFilter(RequestIdFilter())

    # Silence noisy third-party libraries by default
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)