"""Fixed-load performance with separate warmup and measurement snapshots."""
import hashlib
import copy
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from runtime import (Failure, Parser, boolean, dump, input_path, integer, number,
                     public_args, run_main, tokenizer_info, validate_url, aisbench_info)
from dataset_generator import prepare_data
from prefix_metrics import capture, compare, require_valid, settle, choose_profile
from result_mapper import map_performance
from reports import write_reports
from running_monitor import RunningMonitor


def ratio(value):
    if "%" in value[:-1]:
        raise Failure("repeat_rate 只能是 0..1，或带一个 % 的百分数")
    result = number(0, 100 if value.endswith("%") else 1)(value.rstrip("%"))
    return result / 100 if value.endswith("%") else result


def parser():
    p = Parser(description="AISBench 性能与 prefix 命中率测试", allow_abbrev=False)
    public_args(p)
    p.add_argument("--dataset_type", choices=("normal", "prefix_cache"), default="prefix_cache")
    p.add_argument("--dataset")
    p.add_argument("--warmup_dataset")
    p.add_argument("--input_len", type=integer(4), default=2048)
    p.add_argument("--output_len", type=integer(), default=128)
    p.add_argument("--data_num", type=integer())
    p.add_argument("--test_type", choices=("stream", "text"), default="stream")
    p.add_argument("--repeat", type=integer(), default=1)
    p.add_argument("--prefix_num", type=integer(1, 100000), default=1)
    p.add_argument("--repeat_rate", type=ratio, default=0.5)
    p.add_argument("--prefix_test", type=boolean)
    p.add_argument("--dp", type=integer(1, 100000), default=1)
    p.add_argument("--metrics_endpoint", action="append")
    p.add_argument("--pod_info", nargs="+")
    p.add_argument("--metrics_profile", choices=("auto", "ucm_connector", "vllm_prefix", "vllm_prompt"), default="auto")
    p.add_argument("--metrics_policy", choices=("required", "best_effort"), default="best_effort")
    p.add_argument("--metrics_timeout_seconds", type=number(0, 30, open_min=True), default=5)
    p.add_argument("--settle_seconds", type=number(0, 60), default=5)
    p.add_argument("--data_source", choices=("random", "corpus"), default="random")
    p.add_argument("--prefix_mode", choices=("shared", "private"), default="shared")
    p.add_argument("--rounds")
    p.add_argument("--test_name", default="")
    p.add_argument("--npu_num", type=integer())
    p.add_argument("--warmup_concurrency", type=integer(1, 100000))
    p.add_argument("--warmup_output_len", type=integer(), default=1)
    p.add_argument("--length_mean", type=integer(4))
    p.add_argument("--length_std", type=number(0))
    p.add_argument("--length_min", type=integer(4))
    p.add_argument("--length_max", type=integer(4))
    p.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    p.add_argument("--monitor_interval_seconds", type=number(0, 60), default=15)
    return p


def validate(args):
    if args.metrics_endpoint and args.pod_info:
        raise Failure("metrics_endpoint 与 pod_info 只能选一个")
    if args.pod_info:
        args.metrics_endpoint = []
        for pod in args.pod_info:
            address = pod if "://" in pod else "http://" + pod
            parts = urlsplit(address)
            args.metrics_endpoint.append(urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/metrics", "", "")))
        args.pod_info = None
    if not args.metrics_endpoint:
        parts = urlsplit(validate_url(args.url))
        args.metrics_endpoint = [urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1") + "/metrics", "", ""))]
    args.metrics_endpoint = [validate_url(value, metrics=True) for value in args.metrics_endpoint]
    if len(args.metrics_endpoint) != len(set(args.metrics_endpoint)):
        raise Failure("metrics_endpoint 不能重复")
    if args.prefix_test is None:
        args.prefix_test = args.dataset_type == "prefix_cache"
    if args.dataset_type == "normal" and args.prefix_test:
        raise Failure("normal 模式不能开启 prefix_test")
    if (args.dataset or args.dataset_type == "normal") and args.explicit & {"--repeat_rate", "--prefix_num"}:
        raise Failure("已有 dataset 或 normal 模式不接受 repeat_rate / prefix_num")
    if args.dataset and "--input_len" in args.explicit:
        raise Failure("已有 dataset 不接受 input_len")
    if args.dataset and args.prefix_test and not args.warmup_dataset:
        raise Failure("已有前缀数据开启预热时需要 warmup_dataset")
    if args.warmup_dataset and not (args.dataset and args.prefix_test):
        raise Failure("warmup_dataset 只能与已有前缀数据和 prefix_test=true 一起使用")
    if not args.dataset:
        args.data_num = args.data_num or 64
        if args.prefix_num > args.data_num:
            raise Failure("prefix_num 不能大于 data_num")
    if (args.length_mean is None) != (args.length_std is None) or (args.length_min is None) != (args.length_max is None):
        raise Failure("length_mean/std 和 length_min/max 必须分别成对提供")
    if args.length_min is not None and args.length_min > args.length_max:
        raise Failure("length_min 不能超过 length_max")
    if (args.dataset or args.data_source == "corpus") and any(getattr(args, key) is not None for key in ("length_mean", "length_std", "length_min", "length_max")):
        raise Failure("变长参数仅用于随机生成数据")
    if args.data_source == "corpus" and args.prefix_mode != "shared":
        raise Failure("private 模式使用 random 数据源")


def work(execution, args):
    if not args.rounds:
        return work_round(execution, args)
    try:
        configs = json.loads(args.rounds) if args.rounds.lstrip().startswith("[") else json.loads(Path(args.rounds).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise Failure(f"rounds 读取失败：{exc}") from exc
    allowed = {"input_len", "output_len", "data_num", "concurrency", "request_rate", "prefix_num", "repeat_rate", "dp", "seed", "test_name", "length_mean", "length_std", "length_min", "length_max", "prefix_mode", "warmup_concurrency", "warmup_output_len"}
    if not isinstance(configs, list) or not configs:
        raise Failure("rounds 必须是非空 JSON 数组")
    actions = {a.dest: a for a in parser()._actions}
    planned = []
    for overrides in configs:
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise Failure("rounds 含不支持的覆盖字段")
        current = copy.deepcopy(args)
        current.rounds = None
        for key, value in overrides.items():
            action = actions[key]
            if value is None or isinstance(value, (bool, list, dict)):
                raise Failure(f"rounds.{key} 类型不正确")
            try:
                parsed = action.type(str(value)) if action.type else str(value)
            except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
                raise Failure(f"rounds.{key}: {exc}") from exc
            if action.choices and parsed not in action.choices:
                raise Failure(f"rounds.{key} 不支持此值")
            setattr(current, key, parsed)
            current.explicit.add("--" + key)
        validate(current)
        planned.append(current)
    root = execution.output
    summaries = []
    try:
        for group, current in enumerate(planned, 1):
            execution.output = root / f"groups/{group}"
            (execution.output / "logs").mkdir(parents=True)
            current.round_offset = (group - 1) * args.repeat
            try:
                work_round(execution, current)
            finally:
                summaries.append(copy.deepcopy(execution.result["metadata"]))
    finally:
        execution.output = root
        execution.result["metadata"]["groups"] = summaries
        execution.result["metadata"]["metrics_complete"] = all(s.get("metrics_complete", False) for s in summaries)
        from reports import merge_reports
        merge_reports(root)


def work_round(execution, args):
    validate(args)
    versions = aisbench_info()
    execution.result["metadata"].update(versions)
    tokenizer, tokenizer_files = tokenizer_info(input_path(args.model_path, directory=True))
    rows, warmup, prefix_count, evidence = prepare_data(args, tokenizer, execution.output)
    args.actual_prefix_count = prefix_count
    variable = bool(args.dataset or args.length_mean is not None or args.length_min is not None)
    execution.result["workload"] = dict(model_name=args.model_name, dataset=args.dataset or "generated-" + args.dataset_type,
                                         input_len="variable" if variable else args.input_len,
                                         output_len=args.output_len, concurrency=args.concurrency)
    metadata = execution.result["metadata"]
    metadata.update(versions, tokenizer_files=tokenizer_files,
                    template_sha256=hashlib.sha256(b"{question}").hexdigest(), metrics_profile=args.metrics_profile,
                    metrics_complete=True, external_lookup_observed=False, attribution="unverified",
                    warmup_coverage="unverified", freshness="unverified", repeat_cache_policy="preserve",
                    seed_support="requested_unverified", ignore_eos_support="requested_check_output_lengths", rounds=[])
    dump(execution.output / "raw/resolved.json", dict(versions, **evidence, tokenizer_files=tokenizer_files,
                                                      parameters={k: v for k, v in vars(args).items() if k != "explicit"}))
    for local_round in range(1, args.repeat + 1):
        round_id = getattr(args, "round_offset", 0) + local_round
        round_dir = execution.output / f"raw/round-{round_id}"
        round_dir.mkdir()
        phases = ["warmup", "measurement"] if args.prefix_test and warmup else ["measurement"]
        round_evidence = {"round": round_id, "phases": {}}
        prefix_summaries = {}
        metadata["rounds"].append(round_evidence)
        for phase in phases:
            warm = phase == "warmup"
            concurrency = (args.warmup_concurrency or (args.concurrency if args.prefix_mode == "private" else prefix_count if args.dp == 1 else args.dp)) if warm else args.concurrency
            output_len = args.warmup_output_len if warm else args.output_len
            labels = dict(round=round_id, phase=phase, concurrency=concurrency,
                          input_len="variable" if variable or warm else args.input_len,
                          output_len=output_len, dataset_type=args.dataset_type)
            metrics_dir = round_dir / "metrics"
            before = capture(args.metrics_endpoint, args.model_name, args.metrics_timeout_seconds, metrics_dir, phase + "-before")
            selected_profile = choose_profile(before, args.metrics_profile)
            selected_args = copy.copy(args)
            selected_args.metrics_profile = selected_profile
            try:
                if not warm:
                    require_valid(before, selected_args, labels)
            except Failure:
                metadata["metrics_complete"] = False
                metadata["external_lookup_observed"] = None
                _, missing, _ = compare(before, before, selected_profile, labels)
                write_reports(execution.output, round_dir / phase, args, labels, [], missing, selected_profile,
                              {}, len(warmup if warm else rows), True)
                raise
            directory = round_dir / phase
            with RunningMonitor(args, metrics_dir, phase) as monitor:
                code = execution.run_aisbench(directory, args, execution.output / "raw/datasets" / phase, "{question}", perf=True,
                                             concurrency=concurrency if warm else None, output_len=output_len)
            settle(args.settle_seconds)
            after = capture(args.metrics_endpoint, args.model_name, args.metrics_timeout_seconds, metrics_dir, phase + "-after")
            cache_metrics, cache_summary, complete = compare(before, after, selected_profile, labels)
            execution.result["metrics"].extend(cache_metrics)
            metadata["metrics_complete"] = metadata["metrics_complete"] and complete
            dump(round_dir / (phase + "-prefix-summary.json"), cache_summary)
            prefix_summaries[phase] = cache_summary
            dump(round_dir / "prefix-summary.json", prefix_summaries)
            external = cache_summary["ucm" if selected_profile == "ucm_connector" else "external"]
            if not warm:
                if not external["complete"]:
                    metadata["external_lookup_observed"] = None
                elif metadata["external_lookup_observed"] is not None:
                    metadata["external_lookup_observed"] |= external["hits"] > 0
            perf_metrics, perf_evidence, error = [], {}, None
            try:
                perf_metrics, perf_evidence, error = map_performance(directory, len(warmup if warm else rows), labels, args.test_type == "stream")
                execution.result["metrics"].extend(perf_metrics)
                round_evidence["phases"][phase] = perf_evidence
                perf_evidence["metrics_profile"] = selected_profile
                perf_evidence["running_requests"] = monitor.summary
                from runtime import metric
                for endpoint, sampled in monitor.summary.items():
                    execution.result["metrics"].append(metric("running_requests", sampled["max"], "count", "max",
                                                              dict(labels, instance=endpoint), sampled["samples"],
                                                              "定时采样最大值，不是连续时间峰值"))
                dump(round_dir / "performance.json", round_evidence)
            except Failure as exc:
                error = exc
                if code:
                    error = Failure(f"AISBench 退出码 {code}，见 {phase}/aisbench.log", "warmup_failed" if warm else "benchmark_failed", 1)
            write_reports(execution.output, directory, args, labels, perf_metrics, cache_summary, selected_profile,
                          monitor.summary, len(warmup if warm else rows), bool(code or error or (not warm and not complete and args.metrics_policy == "required")))
            if code or error:
                raise error or Failure(f"AISBench 退出码 {code}", "warmup_failed" if warm else "benchmark_failed", 1)
            if args.request_rate == 0 and not warm and perf_evidence["measured_concurrency"] < args.concurrency * 0.8:
                execution.result["notes"] += f"第 {round_id} 轮实际并发未达到目标的 80%；不能据此判断并发上限。"
            if any(length != output_len for length in perf_evidence["output_lengths"]):
                execution.result["notes"] += f"第 {round_id} 轮 {phase} 的实际输出长度与设定不同，请核对服务的 ignore_eos 支持。"
            if not warm and not complete and args.metrics_policy == "required":
                raise Failure("缓存计数缺失、重置或不一致；性能和原始快照已保留", "metrics_unavailable", 1)
    execution.result["notes"] += "缓存计数可能受其他流量和刷新延迟影响；命中不等于已完成 UCM 加载。"


if __name__ == "__main__":
    raise SystemExit(run_main("aisbench-perf", parser, work))
