# Build context: runners/aisbench-perf
FROM ghcr.io/aisbench/aisbench_benchmark:v3.1-20260330-master_x86_64_py_310
USER root
WORKDIR /runner
COPY build/requirements.txt /runner/build/requirements.txt
# Install only at build time; the source archive is pinned in requirements.txt.
RUN python3 -m pip install --no-cache-dir -r /runner/build/requirements.txt
COPY . /runner
RUN chmod 0755 /runner/run.sh /runner/run-local.sh \
 && python3 -c "from testplane_model import Chat" \
 && mkdir -p /workspace/output /data /models /benchmark/ais_bench/datasets \
 && chown 10001:10001 /workspace/output
ENV HOME=/tmp \
    AIS_BENCH_DATASETS_CACHE=/benchmark \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/hf \
    HF_DATASETS_CACHE=/tmp/hf/datasets \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/runner \
    LANG=C.UTF-8
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/runner/run.sh"]
