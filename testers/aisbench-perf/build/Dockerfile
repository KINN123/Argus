# Build context: testers/aisbench-perf
FROM ghcr.io/aisbench/aisbench_benchmark:v3.1-20260330-master_x86_64_py_310
USER root
WORKDIR /tester
COPY build/requirements.txt /tester/build/requirements.txt
# Install only at build time; the archive and critical source files are checked.
RUN python3 -m pip install --no-cache-dir -r /tester/build/requirements.txt
COPY . /tester
RUN chmod 0755 /tester/run.sh /tester/run-local.sh \
 && TESTPLANE_AISBENCH_CHECK=strict python3 -c "from runtime import verify_aisbench; verify_aisbench(); from testplane_model import Chat" \
 && mkdir -p /workspace/output /data /models \
 && chown 10001:10001 /workspace/output
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/hf \
    HF_DATASETS_CACHE=/tmp/hf/datasets \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/tester \
    LANG=C.UTF-8
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/tester/run.sh"]
