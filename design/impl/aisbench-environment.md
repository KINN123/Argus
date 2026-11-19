# AISBench 现有容器的环境检查

覆盖范围：`testers/aisbench-acc`、`testers/aisbench-perf` 在已有 Linux 容器中直接运行时的依赖检查、诊断和版本记录。此次版本为 0.1.1。

不覆盖：平台状态机、数据表、API，以及 Tester 向 Argus v2 Runner 契约的迁移。这里沿用两个目录现有的命令行入口和结果结构，不能据此认为它们已经完成 v2 平台接入。

## 为什么需要调整

同样标记为 AISBench 3.1 的镜像，源码提交或本地修改可能不同。原实现只要核心文件的 SHA256 与固定提交不一致就退出，导致已有环境在实际执行之前被拦下。哈希不同不能单独证明接口不兼容。

## 两种检查方式

通过环境变量 `TESTPLANE_AISBENCH_CHECK` 选择，两个 Tester 使用相同实现。

| 值 | 行为 |
|---|---|
| `strict`，默认 | 保持核心文件哈希检查，只有与 `build/aisbench-lock.json` 一致才继续。 |
| `compatible` | 允许 AISBench 3.1 系列的核心文件哈希不同，打印提示并记录差异，再继续实际测试。 |

无效取值退出 2。compatible 只放宽哈希差异，不忽略 AISBench 包不可用、必要文件缺失、依赖缺失、锁文件损坏，也不接受其他版本系列。3.1 系列使用 Python 包版本的 release 前两段判断，允许 3.1、3.1.0 和同系列预发布/本地版本，拒绝 3.10。

检查与执行都使用 `run.sh` 调用的同一个 `python3` 环境。缺包提示须包含当前解释器，不能一律建议更换镜像。两个 `run-local.sh` 将该变量传入 Docker；镜像构建仍使用固定依赖及 strict 检查。

## 保留结果校验与环境证据

compatible 不是全面兼容承诺。实际调用、模板读取、逐题评分、请求数、token 数及性能结果格式仍按原逻辑检查，不兼容就失败，不生成虚假的成功结果。

`result.json` 的 metadata 和成功准备后的 `raw/resolved.json` 记录：

- `aisbench_version`、`python_version`、`transformers_version`：实际环境版本。
- `aisbench_check_mode`：所选检查方式。
- `aisbench_source_verified`：锁文件内的核心文件是否全部匹配。
- `aisbench_source_mismatches`：不匹配文件的相对路径列表。
- `aisbench_expected_commit`：适配基线提交。
- `aisbench_commit`：核心文件匹配时沿用基线提交；不匹配时为 null，不能冒充正在使用固定提交。

环境检查通过后立即保存 metadata，即使后续 tokenizer 或模板检查失败，也保留上述证据。非严格源码的提示写入终端及 `logs/runner.log`。

## 验证

离线测试覆盖 strict 拒绝差异、compatible 接受 3.1 差异并如实记录、3.10/其他系列拒绝、缺文件/缺包/坏锁仍失败。保留现有精度和性能结果校验用例。

已有 Linux 容器可先运行各目录 `tests/real_smoke.py`：用本机模拟 HTTP 服务验证已安装的真实 AISBench 调用与结果解析，不需要真实模型和数据集。该检查通过后，再用真实服务、少量数据运行两个 `run.sh`。本地离线测试不能代替用户镜像和 UCM 服务联调。

2026-11-19 本地验证：新增环境检查每个 Tester 10 项，全部通过；两套离线测试共 71 项，65 项通过。其余 6 项在修改前就因旧结果结构不满足当前 OpenAPI 的 `attempt_no` 必填要求而失败，属于尚未完成的 v2 契约迁移。本次没有修改这些用例或跳过校验，也尚未运行真实 AISBench 镜像。
