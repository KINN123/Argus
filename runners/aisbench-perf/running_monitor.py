"""Sample running requests separately for each phase and endpoint."""
import json
import os
import threading
from urllib.request import Request, build_opener

from prometheus_client.parser import text_string_to_metric_families

from prefix_metrics import NoRedirect
from runtime import dump, now, redact


class RunningMonitor:
    def __init__(self, args, directory, phase):
        self.args, self.directory, self.phase = args, directory, phase
        self.stop_event = threading.Event()
        self.summary = {}
        self.thread = None

    def __enter__(self):
        if self.args.monitor_interval_seconds:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        dump(self.directory / (self.phase + "-running-summary.json"), self.summary)

    def run(self):
        import math
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / (self.phase + "-running-samples.jsonl")).open("w", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                for endpoint in self.args.metrics_endpoint:
                    if self.stop_event.is_set():
                        break
                    row = dict(timestamp=now(), endpoint=endpoint, value=None)
                    try:
                        headers = {"Accept": "text/plain"}
                        if os.environ.get("METRICS_API_KEY"):
                            headers["Authorization"] = "Bearer " + os.environ["METRICS_API_KEY"]
                        with build_opener(NoRedirect()).open(Request(endpoint, headers=headers), timeout=self.args.metrics_timeout_seconds) as response:
                            text = response.read(16 * 1024 * 1024 + 1)
                        if len(text) > 16 * 1024 * 1024:
                            raise ValueError("metrics response too large")
                        values = [sample.value for family in text_string_to_metric_families(text.decode("utf-8"))
                                  for sample in family.samples if sample.name == "vllm:num_requests_running"
                                  and all(sample.labels.get(k, self.args.model_name) == self.args.model_name for k in ("model", "model_name"))]
                        if values and all(math.isfinite(v) and v >= 0 for v in values):
                            row["value"] = sum(values)
                            summary = self.summary.setdefault(endpoint, dict(max=None, samples=0))
                            summary["max"] = max(summary["max"] or 0, row["value"])
                            summary["samples"] += 1
                    except Exception as exc:
                        row["error"] = redact(str(exc))
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                self.stop_event.wait(self.args.monitor_interval_seconds)
