# aisbench-acc

给一个已经运行的 OpenAI 兼容模型服务，用 AISBench 做 GSM8K 精度测试。Tester 不启动模型，也不负责开关 UCM。

`run.sh` 接收参数并启动 `runner.py`。Python 检查数据、生成 AISBench 配置、运行完整推理和评分，最后写 `result.json`。第一版只支持以下两种配置：

| dataset_config | dataset 内容 |
|---|---|
| `gsm8k_4shot_cot`，默认 | GSM8K 目录，含 `train.jsonl` 和 `test.jsonl`；使用官方固定的 4 个示例 |
| `gsm8k_jsonl_0shot` | JSONL 文件，每行含 `question` 和数字 `answer`；不带示例 |

0shot 文件可以这样写，标准 GSM8K 的 `#### 2` 答案也支持：

```json
{"question":"What is one plus one?","answer":"2"}
```

100 道题答对 40 道，执行成功并报告 40%。请求失败或缺少逐题评分，执行失败。`--replay_rounds=2` 会真正再发一轮请求，分别保留结果。

## 构建和本地运行

需要 Linux Docker，当前构建目标是 `linux/amd64`。构建上下文是本 Tester 目录：

```bash
docker build --platform linux/amd64 \
  -f testers/aisbench-acc/build/Dockerfile \
  -t ghcr.io/ucm-system/testplane-tester-aisbench-acc:0.1.1 \
  testers/aisbench-acc

MODEL_DIR=/absolute/path/to/tokenizer DATA_DIR=/absolute/path/to/data \
  sh testers/aisbench-acc/run-local.sh \
  --url=http://model-server:8000/v1 --model_name=served-model \
  --dataset=/data/gsm8k --max_samples=100 --concurrency=4 \
  --replay_rounds=2
```

`MODEL_DIR` 挂到容器 `/models/model`，脚本自动传 `--model_path=/models/model`；`DATA_DIR` 挂到 `/data`。模型地址必须能从容器访问，本机服务在 Docker Desktop 通常用 `host.docker.internal`。每次输出放在新的 `testers/aisbench-acc/output/日期时间-进程号/`，也可用 `OUTPUT_DIR` 指定新的空目录。

需要鉴权时，提前在 shell 设置 `ENDPOINT_API_KEY` 环境变量。密钥不写进命令行或配置。完整参数见 [tester.yaml](tester.yaml) 或镜像的 `--help`。

已有 Linux Python 环境也能直接运行：先安装 [build/requirements.txt](build/requirements.txt)，再执行 `TESTPLANE_OUTPUT_DIR=/new/empty/output python3 runner.py ...`，自己传全部三个公共参数。AISBench 包含 Linux 专用依赖，不支持直接在 Windows Python 中跑真实评测。

### 保留容器里已有的 AISBench 3.1

如果报错是源码哈希不同，可以保留现有环境，使用 0.1.1 起提供的兼容模式。从容器内的项目根目录执行：

```bash
export TESTPLANE_AISBENCH_CHECK=compatible

# 先用已安装的真实 AISBench 和本机模拟服务检查调用、评分和结果解析。
# 无需真实模型和 GSM8K；使用当前 Python 环境的 transformers/tokenizers。
python3 testers/aisbench-acc/tests/real_smoke.py

# 检查通过后运行真实模型；替换下面地址、模型名和容器内路径。
TESTPLANE_OUTPUT_DIR="/tmp/acc-$(date +%Y%m%d-%H%M%S)-$$" \
sh testers/aisbench-acc/run.sh \
  --url=http://model-server:8000/v1 --model_name=served-model \
  --model_path=/models/model --dataset=/data/gsm8k --max_samples=10
```

默认值仍是 `strict`。compatible 仅放宽 3.1 系列的源码哈希差异，不保证所有 3.1 构建都兼容；缺包、缺文件和实际结果不完整仍会失败。不会安装或修改容器里的 AISBench。`run-local.sh` 也会透传该变量。

实际版本、检查模式和不匹配文件写入 `result.json` 的 metadata；源码不同的 `aisbench_commit` 为 null，不冒充固定提交。提示同时写入 `logs/runner.log`。具体约定见 [环境检查设计](../../design/impl/aisbench-environment.md)。

## 平台接入

平台会从 `tester.yaml` 注册本 Tester；镜像需提前构建并发布到该文件的 image 地址。`url/model_name/model_path` 来自 Run.service，其他参数写在 tests[].args；不用在 args 重复公共参数。

```yaml
run:
  name: gsm8k-ucm-on
  service:
    url: http://model-api:8000/v1
    model_name: served-model
    model_path: /models/served-model
  precheck: auto
  tests:
    - name: accuracy
      tester: aisbench-acc
      version: "0.1.1"
      execution: {profile: k8s-npu}
      timeout_seconds: 7200
      args: >-
        --dataset=/data/gsm8k --max_samples=100
        --concurrency=4 --replay_rounds=2
```

环境方要准备同 namespace 的 `testplane-datasets` 和 `testplane-models` PVC；分别只读挂到 `/data`、`/models`。PVC 名字不符合环境时，在发布前修改 tester.yaml 的 mounts.source，并随版本管理。只有路径参数不会把文件传入容器。

## 结果和检查

输出目录包含 `result.json`、`raw/resolved.json`、`raw/dataset/`、每轮的 `requests.jsonl`、`sample_scores.jsonl`、AISBench 原始目录和日志。任务编号从 `TESTPLANE_RUN_ID/TASK_ID/ATTEMPT_ID` 读取，本地默认 local。退出码 0 表示评测完整；2 表示输入或版本不支持；1 表示执行或结果不完整。

默认运行前会严格检查 [AISBench 源码锁](build/aisbench-lock.json)，也可显式启用上述 compatible 模式。适配基线提交为 `525ed1779f50a8ac068b8638771f633a151e769f`。`runtime.py`、`testplane_model.py`、构建依赖和锁文件与性能 Tester 相同；独立镜像需要各自包含副本，测试会检查它们一致。

在仓库根目录执行离线测试（无需 AISBench 或模型）：

```bash
python -m pip install PyYAML==6.0.2 jsonschema==4.23.0 prometheus-client==0.21.1
python -m unittest discover -s testers/aisbench-acc/tests -v
```

真实 AISBench 与小型模拟 HTTP 服务的集成检查：

```bash
docker run --rm --init --read-only --shm-size=1g --tmpfs /tmp:rw,exec,size=4g \
  --entrypoint python3 ghcr.io/ucm-system/testplane-tester-aisbench-acc:0.1.1 \
  /tester/tests/real_smoke.py
```

CI 会运行离线测试并构建镜像做上述集成检查。它验证请求、评分、第二轮重发和凭据处理，不能替代真实模型/UCM 验收。真实验收仍需同模型同题目分别跑 UCM 关/开环境，并结合性能 Tester 和 UCM 加载记录。

设计依据：[精度设计](../../design/impl/aisbench-acc.md)。本次开发机没有 Docker，容器构建和真实服务运行仍待 Linux 环境验证；不能把离线样例当作实测报告。
