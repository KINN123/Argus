# aisbench-perf

给一个已经运行的 OpenAI 兼容模型服务，在固定输入长度、输出长度和并发下测性能，同时采集 prefix 命中率。它不搜索最大并发，也不启动模型或清缓存。

0.1.4 修复 Transformers 5.2 返回 `BatchEncoding` 时的前缀误判：现在取 `input_ids` 比较 token，不再把两个字典字段当作两个 token。更新完整 Tester 目录后，原来的命令可以继续使用，无需更换 AISBench 镜像。请求记录里的本地 token 数补算也使用同一处理方式。

0.1.3 修复部分 AISBench 3.1 输出对象没有 `data_id` 时记录失败的问题。请同步完整目录，尤其是 `runtime.py`、`testplane_model.py` 和新增的 `testplane_inferencer.py`。预热和正式请求都通过这层传递样本编号；执行失败时终端会显示原始日志末尾。

流程是：准备数据 → 采集计数 → 发前缀预热请求 → 再采计数 → 重新采正式测量前计数 → AISBench 正式压测 → 采集结束计数并计算增量。预热和正式测量分开保存。

数据构造参考 [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix/tree/ac06bd619726663ac8c7b0c84c9b3cdbb041b062)：公共前缀 + 3 个随机 token 的文本 + 后缀。`--repeat_rate=50%` 是目标前缀比例，真实命中率以服务计数为准。生成时会重新编码检查长度和前缀；不使用跨任务共享的 picked_ids 文件。来源和区别见 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 构建和本地运行

需要 Linux Docker，当前构建目标是 `linux/amd64`：

```bash
docker build --platform linux/amd64 \
  -f testers/aisbench-perf/build/Dockerfile \
  -t ghcr.io/ucm-system/testplane-tester-aisbench-perf:0.1.4 \
  testers/aisbench-perf

MODEL_DIR=/absolute/path/to/tokenizer DATA_DIR=/absolute/path/to/data \
  sh testers/aisbench-perf/run-local.sh \
  --url=http://model-server:8000/v1 --model_name=served-model \
  --input_len=2048 --output_len=128 --data_num=64 --concurrency=8 \
  --prefix_num=2 --repeat_rate=50% --repeat=2 \
  --metrics_endpoint=http://prefill-0:8000/metrics \
  --metrics_endpoint=http://prefill-1:8000/metrics \
  --metrics_profile=ucm_connector
```

脚本将 MODEL_DIR 挂到 `/models/model` 并传 model_path；DATA_DIR 挂到 `/data`，自动生成数据时可给任意可读数据目录。输出每次使用新的 `output/日期时间-进程号/`；用 OUTPUT_DIR 可指定新的空目录。容器内不能用 localhost 访问宿主机模型，地址须按实际网络设置。

如果使用已有数据，改成：

```bash
--dataset=/data/requests.jsonl --warmup_dataset=/data/prefixes.jsonl \
--prefix_test=true --output_len=128 --concurrency=8 \
--metrics_endpoint=http://prefill-0:8000/metrics
```

两个 JSONL 都需要非空 question，不能同时显式指定 input_len、repeat_rate 或 prefix_num。正式文件的每条请求需要对应的 token 前缀；已有数据不再按 input_len 截断。不要预热时写 `--prefix_test=false` 并省略 warmup_dataset；正常的数据模式可用 `--dataset_type=normal`。

模型凭据用 `ENDPOINT_API_KEY`，指标凭据用 `METRICS_API_KEY`，两者不混用。参数清单和默认值见 [tester.yaml](tester.yaml) 或 `--help`。已有 AISBench 的 Linux 容器可直接执行 run.sh，通过 TESTPLANE_OUTPUT_DIR 指定输出目录，不必安装本仓固定依赖；真实 AISBench 不支持 Windows 原生运行。

### 在已有 AISBench 容器里运行

从 0.1.2 起取消运行前的兼容性检查，直接使用当前 Python 环境里的 AISBench。从容器内的项目根目录执行：

```bash
# 替换下面地址、模型名和容器内路径。
TESTPLANE_OUTPUT_DIR="/tmp/perf-$(date +%Y%m%d-%H%M%S)-$$" \
sh testers/aisbench-perf/run.sh \
  --url=http://model-server:8000/v1 --model_name=served-model \
  --model_path=/models/model --input_len=256 --output_len=32 \
  --data_num=16 --concurrency=2 --repeat_rate=50% \
  --metrics_endpoint=http://prefill-0:8000/metrics
```

无需设置 `TESTPLANE_AISBENCH_CHECK`；以前设置过也不影响执行。不检查版本范围、源码哈希和内部文件清单，也不增加接口预检。实际调用、结果解析和缓存指标采集错误仍按原逻辑报告。

实际版本尽力写入 `result.json` 的 metadata，读不到则填 null。具体约定见 [现有环境运行说明](../../design/impl/aisbench-environment.md)。可选的 `python3 testers/aisbench-perf/tests/real_smoke.py` 使用本机模拟服务检查流程，不是正式测试的前置条件。

## 指标怎么看

| 指标配置 | HBM 分母 | 外部缓存分母 |
|---|---|---|
| `ucm_connector` | Connector 全部查询 token | UCM 命中 / Connector 全部查询 token |
| `vllm_prefix` | HBM 查询 token | external 命中 / external 查询 token，即 HBM 未命中后的条件比例 |

每台实例先做“结束累计值减开始累计值”，再将全部实例的查询和命中相加后相除。只采一部分实例不能冒充全部结果；计数重置、缺指标、两套别名重复都会标记无效。合法的 0 次查询对应 null，存在查询但没命中才是 0%。无模型标签的 metrics 地址必须由环境方确认只服务目标模型。

默认 `metrics_policy=required`：采集不完整则失败。`best_effort` 可以保留完整性能结果，缺失的命中率为 null；它不会忽略请求失败。PD 场景要填写稳定的 Prefill 实例地址，不用负载均衡网关；不同 DNS 名称指向同一实例也不能重复填写。

标准性能包括 TTFT、TPOT、ITL、端到端时延、请求/token 吞吐、实际并发和请求数。时延来自 AISBench 请求时间点，按线性插值算分位数；吞吐读取 AISBench 服务整体汇总。ITL 按流式间隔计数，网络 chunk 可能包含多个 token。非流式的逐 token 指标为 null；单 token 输出的 TPOT/ITL 为 null。

`repeat` 之间不清缓存，后几轮可能更热。`dp` 只影响预热副本和并发，不能保证路由覆盖；命中也不等于 UCM 已成功加载，报告将 attribution、warmup_coverage、freshness 标为 unverified，需要服务端记录辅助确认。

## 平台和输出

平台自动传入 Run.service 的 url/model_name/model_path，以上其他参数写 tests[].args。Tester 声明 endpoint_exclusive=true；同 endpoint 的平台内任务会受独占调度约束，环境里的其他流量仍需使用方隔离。

需要准备同 namespace 的 `testplane-models`、`testplane-datasets` PVC，或发布前修改 tester.yaml 的 mounts.source。修改后按平台版本规则发布，文件不会因为传了路径自动上传。

输出包括 `result.json`、`raw/resolved.json`、`raw/datasets/`，每轮的 warmup/measurement 请求、AISBench 原始结果、计数快照 `.prom`、采集时间/错误记录和分组命中率。退出码 0 表示所选策略下完整完成；2 为参数/配置问题；1 为请求、执行、结果或 required 采集失败。多轮任一轮失败，已完成结果仍保留。

## 开发检查

在仓库根目录运行：

```bash
python -m pip install PyYAML==6.0.2 jsonschema==4.23.0 prometheus-client==0.21.1
python -m unittest discover -s testers/aisbench-perf/tests -v

docker run --rm --init --read-only --shm-size=1g --tmpfs /tmp:rw,exec,size=4g \
  --entrypoint python3 ghcr.io/ucm-system/testplane-tester-aisbench-perf:0.1.4 \
  /tester/tests/real_smoke.py
```

后一个命令运行真实 AISBench，但模型和缓存计数由模拟 HTTP 服务提供，只验证接入流程。CI 包含构建和此项检查。真实 UCM 验收仍需外部命中、加载记录和目标 PD/DP 环境；本次开发机没有 Docker，这部分不能声称已验证。

本仓镜像的构建基线见 [AISBench 源码锁](build/aisbench-lock.json)，运行时不读取或校验它。独立镜像内的 runtime.py、testplane_model.py 和构建依赖与精度 Tester 保持一致。设计依据：[性能设计](../../design/impl/aisbench-perf.md)。
