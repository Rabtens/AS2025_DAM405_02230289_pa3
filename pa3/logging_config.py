"""
pa3/logging_config.py - structured logging for every script and the service (LO7)
--------------------------------------------------------------------------------
Design
  Logger      one logger per component, all under the "pa3" namespace
              (pa3.train, pa3.explain, pa3.service, pa3.monitor, pa3.drift)
  Handlers    console      -> human-readable,  level from LOG_LEVEL (INFO by default)
              app file     -> logs/<name>.log, JSON lines, INFO+, rotating 5 MB x 5
              error file   -> logs/errors.log, JSON lines, ERROR+ only, rotating
  Format      every record carries an ISO-8601 timestamp, level, logger name and
              message; extra fields passed with  logger.info(..., extra={"fields": {...}})
              are written as top-level JSON keys (request_id, latency_ms, ...),
              so the file can be parsed and queried by log tools without regexes.
  Levels      DEBUG    raw payloads, per-trial detail      (development only)
              INFO     normal events: prediction, run finished, summary
              WARNING  handled but unexpected: unknown category, drift detected
              ERROR    a request or step failed; the process continues
              CRITICAL the process cannot continue (e.g. model cannot be loaded)
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone

from pa3.config import LOG_DIR

_CONFIGURED: set[str] = set()


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy for humans to grep and for tools to parse."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
                                 .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))          # structured extras
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Readable console line that still shows the structured extras."""

    def __init__(self):
        super().__init__("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    def format(self, record):
        line = super().format(record)
        fields = getattr(record, "fields", None)
        if fields:
            line += " | " + " ".join(f"{k}={v}" for k, v in fields.items())
        return line


def setup_logging(log_name: str, level: str | None = None) -> None:
    """Attach console + JSON file + error-file handlers to the 'pa3' root logger.

    log_name chooses the file: logs/<log_name>.log (e.g. training.log, service.log).
    Safe to call more than once; handlers are only added the first time per name.
    """
    if log_name in _CONFIGURED:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("pa3")
    root.setLevel(logging.DEBUG)          # handlers decide what is actually written
    root.propagate = False

    console = logging.StreamHandler()
    console.setLevel((level or os.getenv("LOG_LEVEL", "INFO")).upper())
    console.setFormatter(ConsoleFormatter())

    app_file = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{log_name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    app_file.setLevel((level or os.getenv("LOG_LEVEL", "INFO")).upper())
    app_file.setFormatter(JsonFormatter())

    err_file = logging.handlers.RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    err_file.setLevel(logging.ERROR)
    err_file.setFormatter(JsonFormatter())

    # replace handlers from an earlier setup_logging call with a different name
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (console, app_file, err_file):
        root.addHandler(h)
    _CONFIGURED.clear()
    _CONFIGURED.add(log_name)


def get_logger(component: str) -> logging.Logger:
    """Logger for one component, e.g. get_logger('service') -> 'pa3.service'."""
    return logging.getLogger(f"pa3.{component}")
