# 第三方来源

0.3.0 继续参考用户提供的 reference/K8s-AISBench-Test.Jenkinsfile，增加原生数据集命令、六列 prefix 用例、performance_flat.csv 和旧结果 JSON 结构。来源哈希在 reference-sources.json，边界见 design/impl/aisbench-jenkins-migration.md。

0.3.0 还参考用户在本仓 reference/ais_bench 与 reference/aisbench_auto_tools_prefix 提供的旧实现。迁移的是私有/共享前缀方法、两种指标分母、阶段报告字段、原始 AISBench 表格合并和 running 请求数采样；没有直接执行旧脚本或修改 AISBench 安装目录。源文件 SHA-256 记录在 reference-sources.json，逐项约定见 design/impl/aisbench-reference-migration.md。默认随机词表生成，GSM8K 语料仅用于显式 data_source=corpus。

- 方法参考：rayn-zzz/aisbench_auto_tools_prefix，提交 `ac06bd619726663ac8c7b0c84c9b3cdbb041b062`，MIT，许可证见 [LICENSE.aisbench_auto_tools_prefix](third_party/LICENSE.aisbench_auto_tools_prefix)。参考 generate_dataset.py、aisbench_test.py、cal_prefix_hit_rate.py 的“前缀 + 三 token 分隔 + 后缀、先预热、前后计数做差”方法；本目录重新实现参数解析、生成和采集，没有直接执行其可变 config.py 或 shell 命令。
- 生成语料：[OpenAI GSM8K](https://github.com/openai/grade-school-math)，MIT，许可证见 [LICENSE.GSM8K](third_party/LICENSE.GSM8K)。`data/GSM8K.jsonl` 来自本仓 aisbench-sla 已携带的 GSM8K 文件；内容校验值随每次运行记录。这份生成语料不用于给性能请求评分。
- AISBench：[AISBench/benchmark](https://github.com/AISBench/benchmark/tree/525ed1779f50a8ac068b8638771f633a151e769f)，Apache-2.0，由构建依赖安装，不在此复制它的实现。运行源码与关键依赖固定在 build/requirements.txt、build/aisbench-lock.json。

与参考工具的差异：随机状态只属于当前任务；生成后重新检查 token 长度和聊天模板前缀；保留所有轮次；使用参数数组调用 AISBench；缺指标、计数重置不改成零；按来源完整标签匹配；明确 UCM 与 vLLM external 的不同分母。密钥仅通过环境传给对应客户端。
