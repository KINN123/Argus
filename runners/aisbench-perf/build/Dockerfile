# Build context: runners/aisbench-perf
FROM registry.dev.huawei.com/flash_stor/aisbench_benchmark:v3.1_x86_64_py_310-transformersv5
USER root
WORKDIR /runner
COPY build/requirements.txt /runner/build/requirements.txt
# Preserve the base AISBench/Transformers environment; only fill the metrics dependency.
RUN python3 -c "import prometheus_client" \
 || python3 -m pip install --no-cache-dir --no-deps -r /runner/build/requirements.txt
COPY . /runner
RUN groupadd --gid 10001 aisbench \
 && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin aisbench \
 && chmod 0755 /runner/run.sh /runner/run-local.sh \
 && mkdir -p /workspace/output /data /models /benchmark/ais_bench/datasets \
 && chown 10001:10001 /workspace/output
ENV HOME=/tmp \
    USER=aisbench \
    LOGNAME=aisbench \
    TORCH_HOME=/tmp/torch \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TRITON_CACHE_DIR=/tmp/triton \
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
# Exercise CLI imports with the same user as the running container.
RUN python3 -c "import os, pwd; assert pwd.getpwuid(os.getuid()).pw_name == 'aisbench'; from testplane_model import Chat; from ais_bench.benchmark.cli.workers import WORK_FLOW"
ENTRYPOINT ["/runner/run.sh"]
