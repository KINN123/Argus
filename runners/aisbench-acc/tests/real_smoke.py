"""Run the installed, real AISBench against a tiny local OpenAI-compatible mock.

Execute inside the built Linux image: python3 /runner/tests/real_smoke.py.
This verifies integration, not real model quality, throughput, or UCM loading.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import chat_token_ids


def main():
    with tempfile.TemporaryDirectory(prefix="testplane-smoke-") as temp:
        work = Path(temp)
        vocab = {word: i for i, word in enumerate(["[UNK]", "alpha", "beta", "gamma", "delta", "The", "answer", "is", "2", "user", "assistant"])}
        raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(work / "model")
        counts = {"requests": 0, "queries": 0, "hits": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                if self.path != "/metrics" or self.headers.get("Authorization") != "Bearer smoke-metrics-key":
                    self.send_error(403)
                    return
                with lock:
                    queries, hits = counts["queries"], counts["hits"]
                content = (f'ucm:total_prefix_query_tokens_total{{model_name="m"}} {queries}\n'
                           f'ucm:gpu_hbm_hit_tokens_total{{model_name="m"}} 0\n'
                           f'ucm:ucm_hit_tokens_total{{model_name="m"}} {hits}\nprocess_start_time_seconds 1\n').encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            def do_POST(self):
                if self.path != "/proxy/v1/chat/completions" or self.headers.get("Authorization") != "Bearer smoke-model-key":
                    self.send_error(403)
                    return
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert request["model"] == "m"
                tokens = len(chat_token_ids(tokenizer, request["messages"]))
                count = request["max_tokens"] if request.get("ignore_eos") else 4
                content = " ".join(["2"] * count) if request.get("ignore_eos") else "The answer is 2"
                usage = dict(prompt_tokens=tokens, completion_tokens=count, total_tokens=tokens + count)
                with lock:
                    counts["requests"] += 1
                    counts["queries"] += tokens
                    counts["hits"] += tokens // 2
                if request.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in content.split(" "):
                        payload = dict(choices=[dict(index=0, delta=dict(content=chunk + " "), finish_reason=None)])
                        self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                        self.wfile.flush()
                        time.sleep(.01)
                    payload = dict(choices=[], usage=usage)
                    self.wfile.write(("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode())
                else:
                    data = json.dumps(dict(choices=[dict(message=dict(role="assistant", content=content), finish_reason="stop")], usage=usage)).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        environment = dict(os.environ, ENDPOINT_API_KEY="smoke-model-key", METRICS_API_KEY="smoke-metrics-key",
                           TESTPLANE_OUTPUT_DIR=str(work / "output"), NO_PROXY="127.0.0.1,localhost")
        dataset = work / "data.jsonl"
        dataset.write_text('\n'.join(json.dumps(dict(question=q, answer="2")) for q in ("alpha beta gamma", "alpha beta delta")) + '\n')
        command = [sys.executable, str(ROOT / "runner.py"), f"--url=http://127.0.0.1:{server.server_port}/proxy/v1",
                   "--model_name=m", f"--model_path={work / 'model'}", f"--dataset={dataset}", "--concurrency=2"]
        perf = (ROOT / "prefix_metrics.py").exists()
        if perf:
            warmup = work / "warmup.jsonl"
            warmup.write_text(json.dumps(dict(question="alpha beta")) + '\n')
            command += [f"--warmup_dataset={warmup}", "--output_len=3", "--repeat=2", "--settle_seconds=0",
                        f"--metrics_endpoint=http://127.0.0.1:{server.server_port}/metrics"]
        else:
            command += ["--dataset_config=gsm8k_jsonl_0shot", "--replay_rounds=2", "--output_len=16"]
        try:
            completed = subprocess.run(command, env=environment, timeout=300)
            if completed.returncode:
                for path in (work / "output").rglob("*.log"):
                    print(path, path.read_text(errors="replace")[-12000:])
                raise AssertionError(f"Tester returned {completed.returncode}")
            result = json.loads((work / "output/result.json").read_text())
            assert result["status"] == "succeeded", result
            assert counts["requests"] == (6 if perf else 4), counts
            if perf:
                assert result["metadata"]["metrics_complete"] is True
                assert result["metadata"]["external_lookup_observed"] is True
                assert any(m["name"] == "ttft" and m["value"] is not None for m in result["metrics"])
            else:
                scores = [m["value"] for m in result["metrics"] if m["name"] == "accuracy"]
                assert scores == [100, 100], scores
            for file in (work / "output").rglob("*"):
                if file.is_file():
                    content = file.read_bytes()
                    assert b"smoke-model-key" not in content and b"smoke-metrics-key" not in content, file
            print("PASS: real AISBench + mock HTTP, two rounds, auth and artifact checks")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
