"""AISBench common-result JSON plus its audited per-request time points."""
import json
import math

from runtime import DATA, MODEL, Failure, jsonl, metric


def scalar(value, unit=""):
    if isinstance(value, str):
        if unit and not value.endswith(" " + unit):
            raise Failure("AISBench 指标单位不匹配", "upstream_result_invalid", 1)
        value = value[:-len(unit)].strip() if unit else value
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise Failure("AISBench 指标包含无效数值", "upstream_result_invalid", 1)
    return result


def statistics(values):
    if not values:
        return {stat: None for stat in ("mean", "min", "max", "p50", "p90", "p99")}
    values = sorted(values)
    def quantile(q):
        pos = (len(values) - 1) * q
        lower = int(pos)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (pos - lower)
    return dict(mean=sum(values) / len(values), min=values[0], max=values[-1], p50=quantile(.5), p90=quantile(.9), p99=quantile(.99))


def map_performance(directory, expected, labels, stream):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "requests").glob("*.json"))]
    jsonl(directory / "requests.jsonl", records)
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < expected for i in ids):
        raise Failure("请求 ID 重复或越界", "upstream_result_invalid", 1)
    successes = [r for r in records if r.get("success") is True]
    empty = sum(not isinstance(r.get("prediction"), str) or not r["prediction"].strip() or r["output_tokens"] <= 0 for r in successes)
    counts = dict(total=expected, success=len(successes), failed=expected - len(successes), empty=empty)
    metrics = [metric("requests", value, "count", "count", dict(labels, kind=kind)) for kind, value in counts.items()]
    if len(successes) != expected or empty:
        return metrics, {}, Failure("请求失败、缺失或出现空回答", "incomplete_requests", 1)
    files = list(directory.glob(f"aisbench/**/performances/{MODEL}/{DATA}.json"))
    if len(files) != 1:
        raise Failure("没有唯一的 AISBench 性能汇总文件", "upstream_result_invalid", 1)
    common = json.loads(files[0].read_text(encoding="utf-8"))
    def get(name, unit=""):
        return scalar(common[name]["total"], unit)
    if get("Total Requests") != expected or get("Success Requests") != expected or get("Failed Requests") != 0:
        raise Failure("AISBench 汇总请求数与实际请求不一致", "upstream_result_invalid", 1)
    duration = get("Benchmark Duration", "ms") / 1000
    if duration <= 0:
        raise Failure("正式测试时长无效", "upstream_result_invalid", 1)
    arrays = dict(ttft=[], tpot=[], itl=[], e2e_latency=[], input_tokens=[], output_tokens=[])
    for row in records:
        points = row["time_points"]
        if len(points) < 2 or any(type(p) not in (int, float) or not math.isfinite(p) for p in points) or points != sorted(points):
            raise Failure("AISBench 请求时间点无效", "upstream_result_invalid", 1)
        for key in ("input_tokens", "output_tokens"):
            if type(row[key]) is not int or row[key] <= 0:
                raise Failure("实际 token 数缺失", "upstream_result_invalid", 1)
            arrays[key].append(row[key])
        latency, ttft = points[-1] - points[0], points[1] - points[0]
        arrays["e2e_latency"].append(latency * 1000)
        if stream:
            arrays["ttft"].append(ttft * 1000)
            if row["output_tokens"] > 1:
                arrays["tpot"].append((latency - ttft) / (row["output_tokens"] - 1) * 1000)
                arrays["itl"].extend((b - a) * 1000 for a, b in zip(points[1:], points[2:]))
    for name, values in arrays.items():
        unit = "tokens" if name.endswith("tokens") else "ms"
        for statistic, value in statistics(values).items():
            metrics.append(metric(name, value, unit, statistic, labels, len(values), "此模式或输出长度没有可用的逐 token 时间" if value is None else None))
    input_total, output_total = sum(arrays["input_tokens"]), sum(arrays["output_tokens"])
    if get("Total Input Tokens") != input_total or get("Total Generated Tokens") != output_total:
        raise Failure("AISBench token 汇总与实际请求不一致", "upstream_result_invalid", 1)
    concurrency = get("Concurrency")
    for name, value, unit, stat in (("benchmark_duration", duration, "s", "sum"),
                                  ("measured_concurrency", concurrency, "count", "mean"),
                                  ("request_throughput", get("Request Throughput", "req/s"), "req_per_s", "mean"),
                                  ("input_throughput", get("Input Token Throughput", "token/s"), "tokens_per_s", "mean"),
                                  ("output_throughput", get("Output Token Throughput", "token/s"), "tokens_per_s", "mean"),
                                  ("total_throughput", get("Total Token Throughput", "token/s"), "tokens_per_s", "mean"),
                                  ("input_tokens", input_total, "tokens", "sum"), ("output_tokens", output_total, "tokens", "sum")):
        metrics.append(metric(name, value, unit, stat, labels, expected))
    evidence = dict(measured_concurrency=concurrency, duration_seconds=duration,
                    token_sources=sorted({r["token_source"] for r in records}),
                    output_lengths=arrays["output_tokens"], input_lengths=arrays["input_tokens"],
                    latency_source="AISBench RequestOutput.time_points", percentile="linear interpolation",
                    itl_sample_unit="stream intervals; chunks may contain multiple tokens")
    return metrics, evidence, None
