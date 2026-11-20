"""Reference-compatible flat reports; structured AISBench results remain authoritative."""
import csv
import json
from pathlib import Path

from runtime import DATA, MODEL, Failure, dump, jsonl, now
from result_mapper import scalar


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)


def append_row(output, row):
    path = output / "prefix_bench_result.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    rows.append(row)
    jsonl(path, rows)
    write_csv(output / "prefix_bench_result.csv", rows)


def raw_columns(directory):
    columns = {}
    for suffix in ("json", "csv"):
        files = list(directory.glob(f"aisbench/**/performances/{MODEL}/{DATA}.{suffix}"))
        if len(files) != 1:
            continue
        if suffix == "json":
            value = json.loads(files[0].read_text(encoding="utf-8"))
            columns.update({"aisbench_total:" + k: v["total"] for k, v in value.items() if isinstance(v, dict) and "total" in v})
        else:
            with files[0].open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if "Performance Parameters" in row and "Average" in row:
                        columns["aisbench_avg:" + row["Performance Parameters"]] = row["Average"]
    return columns


def rate_fields(group):
    return dict(queries=group.get("queries"), hits=group.get("hits"),
                hit_rate=group["rate"] / 100 if group.get("rate") is not None else None)


def write_reports(output, directory, args, labels, metrics, cache, profile, running, count, failed):
    def value(name, statistic="mean", kind=None):
        return next((m["value"] for m in metrics if m["name"] == name and m["statistic"] == statistic
                     and (kind is None or m["labels"].get("kind") == kind)), None)
    row = dict(timestamp=now(), round=labels["round"], test_name=args.test_name,
               phase="warmup" if labels["phase"] == "warmup" else "full", status="failed" if failed else "succeeded",
               input_len=labels["input_len"], output_len=labels["output_len"], data_num=count,
               concurrency=labels["concurrency"], request_rate=0 if labels["phase"] == "warmup" else args.request_rate,
               dp=args.dp, prefix_num=args.prefix_num, prefix_mode=args.prefix_mode, data_source=args.data_source,
               actual_prefix_count=getattr(args, "actual_prefix_count", None),
               repeat_rate=args.repeat_rate, seed=args.seed, effective_seed=getattr(args, "effective_seed", args.seed),
               model_path=args.model_path, model_name=args.model_name, url=args.url, npu_num=args.npu_num,
               metrics_profile=profile, metrics_complete=all(g["complete"] for g in cache.values()),
               total_requests=value("requests", "count", "total"), measured_concurrency=value("measured_concurrency"),
               ttft_avg_ms=value("ttft"), ttft_p90_ms=value("ttft", "p90"),
               tpot_avg_ms=value("tpot"), tpot_p90_ms=value("tpot", "p90"),
               benchmark_duration_s=value("benchmark_duration", "sum"),
               output_token_throughput=value("output_throughput"), input_token_throughput=value("input_throughput"),
               total_token_throughput=value("total_throughput"), request_throughput_qps=value("request_throughput"),
               total_input_tokens=value("input_tokens", "sum"), total_output_tokens=value("output_tokens", "sum"))
    row["request_throughput_qpm"] = row["request_throughput_qps"] * 60 if row["request_throughput_qps"] is not None else None
    for target, source in (("single_output_throughput", "output_token_throughput"), ("single_total_throughput", "total_token_throughput")):
        row[target] = row[source] / args.npu_num if args.npu_num and row[source] is not None else None
    row["notes"] = "Missing metrics are null; hit_rate fields are ratios (0..1); sampled running maxima are not continuous peaks."
    row["cache_errors"] = {k: v["errors"] for k, v in cache.items() if v["errors"]}
    identities = sorted({(i["endpoint"], json.dumps(i["labels"], sort_keys=True)) for g in cache.values() for i in g["instances"]})
    row["dp_sources"] = {f"dp{index}": dict(endpoint=endpoint, labels=json.loads(identity)) for index, (endpoint, identity) in enumerate(identities)}
    for name, group in cache.items():
        prefix = "hbm" if name == "hbm" else "ext"
        row[prefix + "_denominator"] = group["denominator"]
        for field, v in rate_fields(group).items():
            row[f"{prefix}_{field}_total"] = v
        by_identity = {(i["endpoint"], json.dumps(i["labels"], sort_keys=True)): i for i in group["instances"]}
        for index, identity in enumerate(identities):
            for field, v in rate_fields(by_identity.get(identity, {})).items():
                row[f"{prefix}_{field}_dp{index}"] = v
    raw = raw_columns(directory)
    for field, source, unit in (("max_concurrency", "Max Concurrency", ""),
                                ("prefill_token_throughput", "Prefill Token Throughput", "token/s")):
        raw_value = raw.get("aisbench_total:" + source)
        try:
            row[field] = scalar(raw_value, unit) if raw_value is not None else None
        except (Failure, TypeError, ValueError):
            row[field] = None
            row["notes"] += f" Unsupported AISBench optional field: {source}."
    row["running_requests"] = running
    append_row(output, row)
    if row["phase"] == "full":
        path = output / "hit_rate.jsonl"
        details = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        for endpoint in args.metrics_endpoint:
            item = {k: row[k] for k in ("round", "test_name", "input_len", "output_len", "concurrency", "status", "metrics_profile")}
            item.update(endpoint=endpoint, num_requests=count, hit_rate=args.repeat_rate * 100,
                        **{"vllm:num_requests_running_max": running.get(endpoint, {}).get("max")})
            for name, group in cache.items():
                prefix = "hbm" if name == "hbm" else "ext"
                selected = [i for i in group["instances"] if i["endpoint"] == endpoint]
                queries = sum(i["queries"] for i in selected) if selected and endpoint not in group["errors"] else None
                hits = sum(i["hits"] for i in selected) if queries is not None else None
                item[prefix + "_hit_rate"] = f"{100 * hits / queries:.2f}%" if queries else None
                item[prefix + "(hit/query)"] = f"{hits:g}/{queries:g}" if queries is not None else None
                item[prefix + "_denominator"] = group["denominator"]
            item.update(raw)
            details.append(item)
        jsonl(path, details)
        write_csv(output / "hit_rate.csv", details)
    dump(directory / "report.json", row)


def merge_reports(root):
    for stem in ("prefix_bench_result", "hit_rate"):
        rows = []
        for path in sorted((root / "groups").glob("*/" + stem + ".jsonl"), key=lambda p: int(p.parent.name)):
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if rows:
            jsonl(root / (stem + ".jsonl"), rows)
            write_csv(root / (stem + ".csv"), rows)
