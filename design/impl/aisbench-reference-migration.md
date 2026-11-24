# AISBench 旧实现迁移（0.2.0）

后续 0.3.0 增加 Jenkins 原生精度、普通性能和六列 prefix 用例入口，见 [Jenkins 迁移说明](aisbench-jenkins-migration.md)。下文保留 0.2.0 的迁移依据。

覆盖范围：把用户提供的 reference 两套性能脚本迁入现有 AISBench 精度、性能 Runner，支持当前平台输入和结果契约。服务部署、启停 UCM、清空服务缓存仍由环境负责；不修改平台状态机、数据表或 API。

## 旧实现实际做了什么

| 来源 | 数据 | 指标口径 | 输出 |
|---|---|---|---|
| reference/ais_bench | 每条请求一个私有预埋前缀；先预埋再压测 | HBM 命中 / prompt_tokens；external 命中 / prompt_tokens | 每个 endpoint 一行，合并 AISBench 性能 JSON 的 total 和 CSV 的 Average；后台采样 running requests 最大值 |
| reference/aisbench_auto_tools_prefix | 随机词表生成共享前缀池，每个前缀预热 dp 次；支持多轮和变长 | HBM 命中 / prefix queries；external 命中 / external queries | warmup、full 各一行 CSV/JSONL，包含配置、延迟、吞吐、per-DP 和汇总命中率 |

两个旧工具都通过 vLLM /metrics 读取计数，不要求新 UCM 的 ucm:* 指标。第一套和第二套 external 的分母不同，不能用同一个名字偷偷替换计算口径。旧工具缺失计数填 0、失败继续出成绩、扫描整个工作目录拼接历史结果等行为不迁入。

## 接入方式

正式实现放 runners/aisbench-acc 和 runners/aisbench-perf，runner.yaml 供现有注册器扫描；testers 下保留 run.sh、run-local.sh 转发入口，已有直接运行命令继续有效。两者版本 0.2.0。

支持 --params-file、AG_PARAMS_FILE、默认 /workspace/input/params.jsonl，按该顺序选文件。每行一个业务参数对象，按序执行，任何一行失败则停止；CLI 明确参数覆盖文件对应项。公共服务参数取 CLI 或 AG_SERVED_URL、AG_SERVED_MODEL_NAME、AG_MODEL_PATH。输出目录优先 AG_OUTPUT_DIR，其次兼容 TESTPLANE_OUTPUT_DIR。结果遵守现有 Result schema 2.0，task_id、attempt_no、runner_seq 来自 AG_*，独立执行为 local/1/1。多组输入以 case 标签区分。

AISBench 继续用当前 Python 环境，通过参数数组启动 CLI；每个阶段独立生成配置、数据和 work-dir，不修改镜像安装目录。保留逐请求记录与失败检测、精度评分。性能读当前阶段的结构化性能 JSON/CSV，保留完整原文件，不用终端表格的列位置猜数值。不开启版本、源码或依赖检查。

## 性能参数与流程

- metrics_profile 支持 auto（默认）、vllm_prefix、vllm_prompt、ucm_connector。auto 在预热后的正式测量基线选择：完整 UCM 计数优先，否则本地 vLLM prefix 计数，否则 prompt 计数；选中后固定到该轮结束，不混算。记录实际名称及分母。
- metrics_policy 默认 best_effort：采不到缓存指标时继续性能测试，缺失项为 null 并注明原因；required 在正式测量前及后要求完整计数。预热不会因尚未初始化的缓存计数被阻断。缺失基线不推定为零。
- metrics_endpoint 可重复给定；兼容 pod_info 的 host:port 列表；未给时从服务 base URL 推导 /metrics。保留 endpoint 和全部源标签，不能将不同节点的 engine=0 当作同一实例。
- data_source 默认 random，兼容 corpus；prefix_mode 默认 shared，private 为每条请求独立前缀。前缀长度由 input_len × repeat_rate 得到；shared 中间放 3 个随机 token，private 按旧版使用空格连接后缀。生成后重编码并检查实际长度与聊天 token 前缀。
- seed=0 使用系统熵并记录本次实际 seed；非零可复现。不承诺随机不同的 seed 一定不碰撞。
- rounds 接受 JSON 数组或文件，覆盖负载、seed、test_name；repeat 仍表示同一组数据重复测量、不清缓存。length_mean/std、length_min/max 支持受限高斯或均匀变长；每条请求对应其真实预热前缀。
- warmup_concurrency 和 warmup_output_len 可配置；默认共享模式按 dp（dp=1 时按前缀数），私有模式按正式并发；默认预热输出 1。
- monitor_interval_seconds 默认 15，后台只采集 vllm:num_requests_running，分阶段、分 endpoint 保存采样和最大值。它是采样观测值，不冒充连续时间的峰值。

## 输出

result.json 保存平台标准指标，命中率单位 percent。prefix_bench_result.csv/jsonl 每个阶段一行，兼容旧工具的 round、phase=warmup/full、负载字段、ttft_avg_ms、tpot_p90_ms、吞吐、hbm/ext_hit_rate_total 和 per-DP 列，命中率仍为 0..1。额外写出指标口径、缺失说明、实际预热参数、实例索引。

hit_rate.csv 每个正式测量 endpoint 一行，保留旧版 endpoint、hbm_hit_rate、hbm(hit/query)、ext_hit_rate、ext(hit/query)、running 最大值，以及当前阶段 AISBench 的 JSON total 和 CSV Average 列。原始列加来源前缀防止重名覆盖。

失败的阶段也保存报告行与原始记录，status=failed，不能把已有零散记录当完整成绩。未知单卡数时不编造单卡吞吐；npu_num 是调用者给出的卡数，不自动探测。per-DP 映射在报告中保留 endpoint/engine/worker 标识；缺指标和无查询均为 null，合法的有查询零命中才为 0。

## 验证

离线验证两套旧指标口径、auto 选择、只有 HBM 时继续执行、预热前缺指标、计数重置、模型过滤、多 endpoint/engine、CSV/JSONL 对齐、多轮与变长随机数据、平台 JSONL 输入、Result schema 2.0、注册声明。真实 AISBench 3.1、真实 tokenizer 与 UCM 服务需要在用户容器验收；本机没有该运行环境。

2026-11-20 验证：精度 36 项、性能 63 项离线测试全部通过；server/internal/registry 包测试通过，包含真实 runner.yaml 解析、placeholder 和多轮输入校验。Git Bash 检查入口脚本语法，并经旧 run.sh 转发执行 --help（本机 python3 别名不可用，检查时临时映射到已有 Python）。未构建或发布镜像，未运行真实 AISBench。
