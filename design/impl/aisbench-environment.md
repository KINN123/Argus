# 使用现有 AISBench 环境

本文记录 0.1.4 的环境适配背景；0.2.0 的当前实现、平台 Runner 接入、指标默认策略和报告格式以 [旧实现迁移](aisbench-reference-migration.md) 为准。下文不定义平台状态机、数据表或 API。

## 直接运行

按用户要求，取消运行前的 AISBench 兼容性检查，直接使用 `run.sh` 调用的当前 `python3` 环境。

- 不检查版本范围、固定提交、源码哈希或固定文件清单。
- 不添加接口导入预检，不要求 `utils/perf_details.py` 等特定内部文件存在。
- 移除 `TESTPLANE_AISBENCH_CHECK` 开关；已有 shell 设置该变量也不影响执行。
- 不自动安装、升级或修改 AISBench；自有镜像不需要安装本仓固定依赖后才能运行。

精度测试实际生成提示词时仍读取已安装 AISBench 的所选 GSM8K 模板。模型调用、推理、评分和性能汇总按现有逻辑执行。实际步骤失败时报告该步骤错误，保留已有日志和产物。

## 环境记录和结果

`aisbench_info()` 只尽力读取 AISBench、Python、transformers 版本，写入结果 metadata 和 `raw/resolved.json`。包版本信息不可读时填 null，不影响启动。不把构建使用的固定提交当作用户环境的实际提交。

保留请求数、逐题评分、吞吐、延迟和缓存计数的结果处理，用于确认本次测试实际完成。

`build/requirements.txt` 仍描述本仓 Dockerfile 的构建依赖；`build/aisbench-lock.json` 只作为构建基线参考，不再被运行代码读取。

## 请求记录与样本编号

部分 AISBench 的 GenInferencer 不会向 RequestOutput 写入 data_id。两个 Tester 使用自己的轻量 GenInferencer 子类，在调用上游 do_request 前取 data["index"]，通过 ContextVar 传给同一异步请求中的 Chat 记录器，结束时恢复上下文；请求发送仍由上游执行。

记录器优先使用该样本编号，缺少上下文时才使用已存在的 output.data_id。不能按响应完成顺序编号，也不能将缺失编号都写成 0，以免并发请求的答案与题目错配。每个阶段使用独立记录目录。

模型调用异常时尽力保存失败记录，并重新抛出原始异常；记录过程自身异常不得覆盖模型调用的原始错误。已有的 success=false、HTTP 错误和完整性检查继续保留。

测试失败时直接在终端显示最近一次 AISBench 的日志路径和末尾 100 行（最多 16000 字符），沿用凭据脱敏。即使 AISBench 顶层退出 0，子任务异常或请求不完整也继续报告失败，不能将空结果当作成功。

## 验证

聊天模板返回值统一通过 `chat_token_ids()` 处理：显式指定 `return_dict=False`；对仍返回映射的实现读取 `input_ids`，对数组或单条批次转为一维整数列表。不得把字典键数当作 token 数。该函数同时用于性能数据前缀比较、实际请求的本地 token 数补算和模拟服务检查。

自动生成阶段和最终校验使用同一个前缀匹配函数，同时检查文本、裸 token 和聊天 token。实际 token 不匹配仍失败，不关闭命中率数据校验；现有用户数据校验前先落盘，便于定位。错误需区分文本不匹配、裸 token 不匹配和聊天 token 共享长度不足。

回归测试确认：没有源码锁或 `perf_details.py`、版本不在原先 3.1 范围、没有包版本元数据，以及旧检查环境变量仍存在时，都不会因环境记录而停止。实际 AISBench 与模型服务是否配合正常，需要在用户镜像运行验证。

请求记录回归覆盖没有 data_id 的输出对象、并发乱序返回、上下文恢复、已有 data_id 的输出对象，以及模型异常与记录异常同时发生时保留原始错误。

已有 6 项 OpenAPI 校验失败属于 Argus v2 结果格式迁移；此次不改结果契约，也不跳过其校验。

2026-11-20（0.1.4）：用户确认容器里的 Transformers 5.2.0 聊天模板返回含 input_ids、attention_mask 的 BatchEncoding，旧代码把两个键名当成 token，导致 128-token 前缀校验失败。离线测试 80 项中 74 项通过，6 项为上述原有 OpenAPI 失败。本次新增 9 项测试均通过，覆盖映射、数组、单条批次、错误形状、请求 token 数补算，以及用户的 256 输入长度、16 条数据、50% 前缀参数；使用模拟 tokenizer 验证，并非真实 Qwen tokenizer 验收。尚未在用户的 AISBench 3.1 容器和真实模型服务中完成端到端验收。
