# AISBench 精度测试（0.2.0）

对已经运行的模型服务做 GSM8K 精度测试，保存每题请求、回答、评分和准确率。低分不代表运行失败；漏题、请求失败或评分不完整才失败。模型与 UCM 由环境负责。

实现放在 runners/aisbench-acc，旧入口 testers/aisbench-acc/run.sh 继续有效。不检查 AISBench 固定版本、源码或内部文件。保留了没有 data_id 时的请求记录修复，以及 BatchEncoding 的 input_ids 处理。

## 直接运行

```bash
OUT="/tmp/acc-$(date +%Y%m%d-%H%M%S)-$$"
TESTPLANE_OUTPUT_DIR="$OUT" sh testers/aisbench-acc/run.sh \
  --url=http://127.0.0.1:7800/v1 --model_name=qwen25 \
  --model_path=/mnt/model/Qwen2.5-7B-Instruct/ \
  --dataset=/benchmark/ais_bench/datasets/gsm8k/ \
  --dataset_config=gsm8k_4shot_cot --max_samples=10 --concurrency=1
cat "$OUT/result.json"
```

gsm8k_4shot_cot 使用已安装 AISBench 的官方四例模板，dataset 是含 train.jsonl/test.jsonl 的目录。gsm8k_jsonl_0shot 使用零样本模板，dataset 是 JSONL 文件，每行含 question 和数字 answer，例如 `{"question":"What is one plus one?","answer":"2"}`。replay_rounds=2 会真正重发一轮，并保存两轮评分。

## 平台和输出

平台扫描 runner.yaml；公共参数来自 AG_SERVED_URL、AG_SERVED_MODEL_NAME、AG_MODEL_PATH。业务参数示例：

```json
{"dataset":"/data/gsm8k","dataset_config":"gsm8k_4shot_cot","max_samples":10,"concurrency":1}
```

输入文件顺序：--params-file > AG_PARAMS_FILE > /workspace/input/params.jsonl，每行一组，CLI 明确参数覆盖对应字段。数据和模型由 Execution Profile 挂载。鉴权用 ENDPOINT_API_KEY。

输出优先 AG_OUTPUT_DIR，其次 TESTPLANE_OUTPUT_DIR，每次用新空目录。result.json 使用 schema 2.0；身份来自 AG_TASK_ID、AG_ATTEMPT、AG_RUNNER_SEQ，独立执行为 local/1/1。

结果包含 result.json、raw/resolved.json、raw/dataset，以及 raw/round-N 中的 requests.jsonl、sample_scores.jsonl、AISBench 原始目录和 aisbench.log。多行输入放 cases/N，根 result.json 用 case 标签区分。失败时显示原始日志末尾。退出码 0 评测完整、1 执行失败、2 输入错误。

## 构建和验证

```bash
docker build -f runners/aisbench-acc/build/Dockerfile \
  -t ghcr.io/ucm-system/argus-runner-aisbench-acc:0.2.0 runners/aisbench-acc
python -m unittest discover -s runners/aisbench-acc/tests
# 在已有 AISBench 环境，用模拟服务检查真实调用链：
python3 runners/aisbench-acc/tests/real_smoke.py
```

run-local.sh 支持 MODEL_DIR、DATA_DIR、OUTPUT_DIR、TESTER_IMAGE。源码锁只描述构建基线，不在运行前校验。离线测试不能替代真实模型/UCM 验收。比较 UCM 开关前后精度时，应保持模型、题目和生成参数相同。

设计见 [迁移说明](../../design/impl/aisbench-reference-migration.md)。
