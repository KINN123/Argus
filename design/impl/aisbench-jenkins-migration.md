# Jenkins AISBench 测试迁移（0.3.0）

覆盖 reference/K8s-AISBench-Test.Jenkinsfile 的 acc、perf、prefix 三条测试路径、参数和产物。平台状态机、表结构、API 不变；Pod、挂载和制品保存沿用平台现有能力。

旧 Jenkins 的 acc 并不限于 GSM8K：AISBENCH_DATASETS 是 AISBench 配置名列表。新增 `datasets` 参数，精度走原生 `--mode all --merge-ds --summarizer example`，普通性能走 `--mode perf --summarizer default_perf`。保留各数据集自己的提示词、预处理和 evaluator，不统一换成 GSM8K。实际可用数据集由容器配置、数据文件以及聊天模型支持的推理方式决定，不保证所有 AISBench 数据集都适用。

两种入口互斥：原有 `dataset` 是 Tester 自己处理的 GSM8K/JSONL 路径；新增 `datasets` 是配置名，接受逗号分隔字符串或 JSON 数组。`config_dir` 可指定原有 aisbench_configs 目录；`working_dir` 设置子进程工作目录，默认调用者当前目录，但不替代 AISBench 的数据根目录。模型配置从容器的 vllm_api_stream_chat 模板加载，仅在本次输出目录生成覆盖配置，不修改安装文件。没有版本或源码校验。

2026-11-26 挂载约定（0.3.1）：平台将数据集根目录挂到 `/benchmark/ais_bench/datasets`，其下按数据集组织，例如 `gsm8k/train.jsonl`、`gsm8k/test.jsonl`。两个镜像设置 `AIS_BENCH_DATASETS_CACHE=/benchmark`，使配置中的 `ais_bench/datasets/gsm8k` 指向该挂载。该变量是原生数据集的查找根目录，与 `/tmp/hf/datasets` 中的 Hugging Face 中间缓存不同。数据不打包进镜像，挂载内容需可被运行用户读取。绝对数据路径仍按配置原值使用；自定义配置不采用 `ais_bench/datasets/...` 布局时需填写正确的绝对路径。本地 run-local.sh 将 DATA_DIR 同时挂到固定路径和兼容路径 /data；直接使用已有容器时需自行设置此环境变量。

原生路径参数：num_prompts、num_warmups、temperature、ignore_eos、output_len、concurrency、request_rate。默认低并发，temperature=0.01、ignore_eos=true 对齐 Jenkins；不改变既有 GSM8K 和生成数据路径的缺省值。显式给出的其它模式参数必须拒绝，不能看似接受却不生效。

prefix 仍用现有生成、预热、正式测量流程。增加 `test_cases` 接收旧六列数组 [input_len, output_len, num_requests, hit_rate百分数, concurrency, seed]，转换为私有前缀 rounds；预热输出 max(1, output_len//10)，并发与正式阶段一致，默认 vllm_prompt 指标口径。PD 分离的 POD_INFO 对应 pod_info/metrics_endpoint，逐 endpoint 保留，不自动发现部署形态。

输出：平台 result.json；原始日志、CSV、预测文件保持在本次 raw/native/aisbench 下；额外 accuracy.csv（原始汇总列）、performance_flat.csv（旧 label_statistic 表头）、legacy_results.json（build/type/status/data/extra_info，build 使用平台 task_id）。多份性能表逐份保存并加 source 防止覆盖。数值保留符号和科学计数法，缺失不填 0。退出非零、无有效汇总或汇总缺成绩均失败，仍留原文件。原生路径依赖 AISBench 自身的请求和评分判定，明确记录验证范围，不声称具备旧 GSM8K 逐题审计。

Jenkins 的 MongoDB 写入改为本地产物，由平台收集，不在 Runner 写外部数据库。EXTRA_INFO 对应 extra_info。镜像由 runner.yaml 固定版本；NFS 和模型/数据路径由平台 profile_mount 配置；不搬入 Jenkins 凭据、节点 IP 或特权 Pod。

参数对应：

| Jenkins | Runner / 平台 |
|---|---|
| SERVER_URL / API_MODEL_NAME / MODEL_FOLDER_NAME | AG_SERVED_URL / AG_SERVED_MODEL_NAME / AG_MODEL_PATH；本地用 url / model_name / model_path |
| AISBENCH_DATASETS | datasets 数组；CLI 可用逗号分隔 |
| max_out_len / batch_size | output_len / concurrency |
| num_prompts / num_warmups / request_rate | 同名参数，前两项用于原生数据集路径 |
| temperature / ignore_eos | 同名参数，原生路径沿用 Jenkins 缺省值 |
| TEST_TYPE=acc | aisbench-acc + datasets |
| TEST_TYPE=perf | aisbench-perf + datasets |
| TEST_TYPE=prefix / TEST_CASES | aisbench-perf + test_cases |
| PD_DISAGGREGATION / POD_INFO | 调用者按部署填写 pod_info 或 metrics_endpoint，不额外增加布尔开关 |
| EXTRA_INFO / BUILD_NAME | extra_info / 平台任务名称（prefix 也保存 extra_info） |
| CUSTOM_TEST_IMAGE / Pod YAML | Runner 镜像版本与 Execution Profile |

旧 Check Endpoint 发的 ping 请求不另行复制到 Runner；服务可用性沿用平台执行前检查，本地直接运行由实际 AISBench 请求报告连接错误。Jenkins 分支扫描、构建显示名、MongoDB 写入和归档阶段由平台职责替代，不嵌入测试脚本。

验证包括原生命令构造、参数互斥、多数据集汇总、失败保留、旧六列测试用例转换、结果契约与注册参数。真实 AISBench 3.1 和用户数据集还需在容器运行；本机的离线验证不能代替真实测试。

2026-11-24：精度 44 项、性能 74 项离线测试通过；Go registry 包测试通过。使用临时目录中的 MMEngine 0.10.7 验证生成配置可 Config.fromfile 加载并 Config.dump 保存，另用缓存的真实 AISBench 模型模板验证后处理得以保留（模型类与后处理函数使用替身，不发模型请求）。真实服务、镜像构建与平台任务提交未在本机执行。

原生 CLI 配置选择与汇总格式还对照了 [AISBench ConfigManager](https://github.com/AISBench/benchmark/blob/525ed1779f50a8ac068b8638771f633a151e769f/ais_bench/benchmark/cli/config_manager.py) 和 [模型模板](https://github.com/AISBench/benchmark/blob/525ed1779f50a8ac068b8638771f633a151e769f/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py)。运行时仍使用容器自己的模板，这些来源不构成版本门禁。
