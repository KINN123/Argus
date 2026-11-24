"""Jenkins acc/perf commands, using the container's dataset configs and evaluators."""
import copy
import csv
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

from runtime import Failure, boolean, dump, input_path, integer, metric, aisbench_info


def arguments(parser):
    parser.add_argument("--datasets", help="AISBench 配置名，逗号分隔或 JSON 数组；与 dataset 互斥")
    parser.add_argument("--config_dir", help="包含 datasets、summarizers 子目录的配置根目录")
    parser.add_argument("--working_dir", help="原生 AISBench 工作目录，默认当前目录")
    parser.add_argument("--num_prompts", type=integer(0), default=0)
    parser.add_argument("--num_warmups", type=integer(0), default=0)
    parser.add_argument("--ignore_eos", type=boolean)
    parser.add_argument("--extra_info", default="")


def dataset_names(value):
    try:
        names = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
    except ValueError as exc:
        raise Failure(f"datasets JSON 无效：{exc}") from exc
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", n.strip()) for n in names):
        raise Failure("datasets 需要配置名列表，例如 gsm8k_gen_4_shot_cot_str；不接受路径或命令选项")
    names = [n.strip() for n in names]
    if len(names) != len(set(names)):
        raise Failure("datasets 配置名不能重复")
    return names


def reject_native_only(args, allow_extra=False):
    unused = args.explicit & {"--config_dir", "--working_dir", "--num_prompts", "--num_warmups", "--ignore_eos", "--extra_info"}
    if allow_extra:
        unused.discard("--extra_info")
    if unused:
        raise Failure("这些参数需要 datasets 原生路径：" + ", ".join(sorted(unused)))


def configured_models(settings):
    """Called by AISBench's generated model config; retain upstream postprocessors."""
    from mmengine.config import Config
    root = Path(importlib.import_module("ais_bench.benchmark").__file__).parent
    source = root / "configs/models/vllm_api/vllm_api_stream_chat.py"
    models = copy.deepcopy(Config.fromfile(str(source))["models"])
    for model in models:
        model.update(path=settings["model_path"], model=settings["model_name"], url=settings["url"],
                     max_out_len=settings["output_len"], batch_size=settings["concurrency"],
                     request_rate=settings["request_rate"])
        model["api_key"] = os.environ.get("ENDPOINT_API_KEY", "")
        model.setdefault("generation_kwargs", {}).update(settings["generation"])
    return models


def prepare_command(directory, args, perf):
    allowed = {"--url", "--model_name", "--model_path", "--datasets", "--config_dir", "--working_dir",
               "--num_prompts", "--num_warmups", "--ignore_eos", "--extra_info", "--temperature",
               "--output_len", "--concurrency", "--request_rate", "--seed", "--thinking"}
    if not perf:
        allowed.add("--top_p")
    unused = args.explicit - allowed
    if unused:
        raise Failure("datasets 原生路径不接受这些参数：" + ", ".join(sorted(unused)))
    names = dataset_names(args.datasets)
    cwd = input_path(args.working_dir or str(Path.cwd()), directory=True)
    configs = directory / "configs"
    (configs / "models").mkdir(parents=True)
    if args.config_dir:
        root = input_path(args.config_dir, directory=True)
        # Preserve imports and relative references without touching the installed configs.
        for name in ("datasets", "dataset_collections", "summarizers"):
            if (root / name).is_dir():
                (configs / name).symlink_to(root / name, target_is_directory=True)
    generation = dict(temperature=args.temperature if "--temperature" in args.explicit else 0.01,
                      ignore_eos=True if args.ignore_eos is None else args.ignore_eos)
    if "--seed" in args.explicit:
        generation["seed"] = args.seed
    if "--top_p" in args.explicit:
        generation["top_p"] = args.top_p
    if args.thinking != "default":
        generation["chat_template_kwargs"] = dict(enable_thinking=args.thinking == "on")
    settings = dict(model_path=args.model_path, model_name=args.model_name, url=args.url,
                    output_len=args.output_len, concurrency=args.concurrency, request_rate=args.request_rate,
                    generation=generation)
    model_file = configs / "models/testplane_native_chat.py"
    # MMEngine otherwise treats imported helpers as lazy objects, which cannot be called.
    model_file.write_text("_base_ = []\nfrom native_suite import configured_models\nmodels = configured_models(" + repr(settings) + ")\n", encoding="utf-8")
    command = [sys.executable, "-m", "ais_bench.benchmark.cli.main", "--config-dir", str(configs),
               "--models", "testplane_native_chat", "--datasets", *names,
               "--summarizer", "default_perf" if perf else "example", "--mode", "perf" if perf else "all",
               "--num-warmups", str(args.num_warmups), "--work-dir", str(directory / "aisbench")]
    if args.num_prompts:
        command += ["--num-prompts", str(args.num_prompts)]
    if not perf:
        command.append("--merge-ds")
    dump(directory / "resolved.json", dict(settings, datasets=names, working_dir=str(cwd),
                                           num_prompts=args.num_prompts, num_warmups=args.num_warmups))
    return command, cwd


def numeric(value):
    # Keep signs/exponents; Jenkins' strip-nondigits changed -1 and 1e3 silently.
    match = re.fullmatch(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:[%A-Za-z/][%A-Za-z0-9/ _-]*|\(\d+/\d+\))?\s*", value)
    if match:
        number = float(match[1])
        if math.isfinite(number):
            return number
    return None


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(directory, output, perf):
    rows, metrics, errors = [], [], []
    for path in sorted((directory / "aisbench").rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            table = list(csv.reader(handle))
        if not table:
            continue
        headers = [cell.strip() for cell in table[0]]
        source = path.relative_to(directory).as_posix()
        if perf:
            if not headers or headers[0] != "Performance Parameters" or "Average" not in headers:
                continue
            flat = {"source": source}
            for row in table[1:]:
                if len(row) < 3 or not row[0].strip():
                    continue
                for index in range(2, min(len(row), len(headers))):
                    key = row[0].strip() + "_" + headers[index]
                    if key in flat:
                        errors.append(f"重复性能列：{source}/{key}")
                    value = numeric(row[index])
                    flat[key] = "" if value is None else format(value, ".15g")
                    stat = {"Average": "mean", "Min": "min", "Max": "max", "Median": "p50",
                            "P50": "p50", "P90": "p90", "P95": "p95", "P99": "p99"}.get(headers[index])
                    if stat and value is not None:
                        unit = units(row[0].strip(), row[index])
                        metrics.append(metric("aisbench_performance", value, unit, stat,
                                              dict(source=source, parameter=row[0].strip(), stage=row[1].strip())))
            if len(flat) > 1:
                rows.append(flat)
            common_file = path.with_suffix(".json")
            try:
                common = json.loads(common_file.read_text(encoding="utf-8"))
                counts = [numeric(str(common[key]["total"])) for key in ("Total Requests", "Success Requests", "Failed Requests")]
                if any(v is None for v in counts) or counts[0] <= 0 or counts[1] != counts[0] or counts[2] != 0:
                    errors.append(f"性能请求未全部成功：{source} {counts}")
                for key, stages in common.items():
                    value = numeric(str(stages.get("total", ""))) if isinstance(stages, dict) else None
                    if value is not None:
                        stat = "count" if key.endswith("Requests") else "sum" if key.startswith("Total") or key == "Benchmark Duration" else "max" if key == "Max Concurrency" else "mean"
                        metrics.append(metric("aisbench_performance_total", value, units(key, str(stages["total"])), stat, dict(source=source, parameter=key)))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"无法核对性能请求计数：{common_file.name}: {exc}")
        else:
            if "dataset" not in headers or "metric" not in headers or "mode" not in headers:
                continue
            models = [h for h in headers if h not in ("dataset", "version", "metric", "mode", "total_count")]
            for index, row in enumerate(table[1:], 1):
                if not any(cell.strip() for cell in row):
                    continue
                item = dict(zip(headers, (cell.strip() for cell in row)))
                rows.append(item)
                for model in models:
                    value = numeric(item.get(model, ""))
                    if value is None:
                        errors.append(f"精度汇总缺成绩：{source}/{item.get('dataset')}/{model}")
                    else:
                        metrics.append(metric("aisbench_score", value, "score", "mean",
                                              dict(source=source, row=index, dataset=item.get("dataset", ""),
                                                   evaluator_metric=item.get("metric", ""), model=model)))
    write_csv(output / ("performance_flat.csv" if perf else "accuracy.csv"), rows)
    if not rows or not metrics:
        errors.append("没有找到有效的 AISBench 汇总 CSV，请查看 raw/native/aisbench 和 aisbench.log")
    return rows, metrics, errors


def units(parameter, raw):
    for suffix, unit in (("token/s", "tokens_per_s"), ("req/s", "req_per_s"), ("ms", "ms")):
        if raw.strip().endswith(suffix):
            return unit
    if parameter.endswith("Requests") or "Concurrency" in parameter:
        return "count"
    if parameter in ("InputTokens", "OutputTokens", "Total Input Tokens", "Total Output Tokens"):
        return "tokens"
    return "score"


def work(execution, args, perf=False):
    directory = execution.output / "raw/native"
    directory.mkdir(parents=True)
    command, cwd = prepare_command(directory, args, perf)
    execution.result["workload"] = dict(model_name=args.model_name, datasets=dataset_names(args.datasets),
                                         concurrency=args.concurrency, output_len=args.output_len)
    execution.result["metadata"].update(aisbench_info(), execution_path="aisbench_native",
                                         validation_scope="aisbench_exit_and_summary; no_per_request_audit")
    execution.log("使用容器内 AISBench 数据集配置，结果保留原始评分口径")
    code = execution.run_command(directory, command, cwd)
    rows, metrics, errors = collect(directory, execution.output, perf)
    failed = bool(code or errors)
    legacy_results(execution, args, rows, "perf" if perf else "acc", failed)
    execution.result["metadata"].update(summary_errors=errors)
    if failed:
        raise Failure(f"AISBench 退出码 {code}；" + "；".join(errors), "benchmark_failed", 1)
    execution.result["metrics"].extend(metrics)
    execution.result["notes"] = "原生数据集评分由 AISBench 完成；score 保留原始量纲，不能跨 evaluator 直接比较。"


def legacy_results(execution, args, rows, kind, failed):
    try:
        extra = json.loads(args.extra_info)
    except ValueError:
        extra = args.extra_info
    execution.result["metadata"]["extra_info"] = extra
    dump(execution.output / "legacy_results.json", [dict(build=execution.result["task_id"], type=kind,
                                                        status="FAILURE" if failed else "SUCCESS", data=row, extra_info=extra) for row in rows])
