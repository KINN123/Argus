# 使用现有 AISBench 环境

覆盖范围：两个 AISBench Tester 在用户已有 Linux 容器里直接执行，版本 0.1.2。不涉及平台状态机、数据表、API，也不包含旧 Tester 向 Argus v2 Runner 契约的迁移。

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

## 验证

回归测试确认：没有源码锁或 `perf_details.py`、版本不在原先 3.1 范围、没有包版本元数据，以及旧检查环境变量仍存在时，都不会因环境记录而停止。实际 AISBench 与模型服务是否配合正常，需要在用户镜像运行验证。

已有 6 项 OpenAPI 校验失败属于 Argus v2 结果格式迁移；此次不改结果契约，也不跳过其校验。
