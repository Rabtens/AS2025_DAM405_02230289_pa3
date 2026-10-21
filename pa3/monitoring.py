"""
pa3/monitoring.py - service health signals
-----------------------------------------
Thread-safe counters for the serving process:

  requests          total HTTP requests to /predict
  client_errors     4xx - the request was invalid (bad input)
  server_errors     5xx - the service itself failed
  error_rate        (client + server errors) / requests
  latency           last 1,000 request latencies -> mean, p50, p95, max
  predictions       number of records scored and the share predicted ">50K"
                    (a change in this share is "prediction drift")
  warnings          inputs served despite unknown categories / out-of-range values

summary() writes one structured SUMMARY log line; the service calls it every
SUMMARY_EVERY requests and on GET /metrics. In production these numbers would
be scraped by Prometheus/Grafana; here the log line and endpoint are enough.
"""

import threading
import time
from collections import deque

import numpy as np

from pa3.logging_config import get_logger

log = get_logger("monitor")


class ServiceMonitor:
    def __init__(self, latency_window: int = 1000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = self.client_errors = self.server_errors = 0
        self.records_scored = self.positive_predictions = self.input_warnings = 0
        self.latencies: deque = deque(maxlen=latency_window)

    def record(self, latency_ms: float, status: int, n_records: int = 0,
               n_positive: int = 0, n_warnings: int = 0) -> None:
        with self._lock:
            self.requests += 1
            self.latencies.append(latency_ms)
            if 400 <= status < 500:
                self.client_errors += 1
            elif status >= 500:
                self.server_errors += 1
            self.records_scored += n_records
            self.positive_predictions += n_positive
            self.input_warnings += n_warnings

    def metrics(self) -> dict:
        with self._lock:
            lat = np.array(self.latencies) if self.latencies else np.array([0.0])
            errors = self.client_errors + self.server_errors
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests": self.requests,
                "client_errors": self.client_errors,
                "server_errors": self.server_errors,
                "error_rate": round(errors / self.requests, 4) if self.requests else 0.0,
                "latency_ms_mean": round(float(lat.mean()), 2),
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
                "latency_ms_max": round(float(lat.max()), 2),
                "records_scored": self.records_scored,
                "positive_rate": round(self.positive_predictions / self.records_scored, 4)
                                 if self.records_scored else 0.0,
                "input_warnings": self.input_warnings,
            }

    def summary(self) -> dict:
        m = self.metrics()
        log.info("SUMMARY", extra={"fields": m})
        return m
