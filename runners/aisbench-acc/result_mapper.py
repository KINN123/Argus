"""Check request completeness before accepting AISBench's GSM8K score."""
import json
import math

from runtime import DATA, MODEL, Failure, jsonl, metric


def audit_records(directory, expected):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "requests").glob("*.json"))]
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < expected for i in ids):
        raise Failure("请求记录包含重复或越界 ID", "upstream_result_invalid", 1)
    return sorted(records, key=lambda r: r["id"])


def single_score_detail(detail):
    if not isinstance(detail, dict):
        raise Failure("AISBench 逐题评分必须为对象", "upstream_result_invalid", 1)
    detail = dict(detail)
    # BaseEvaluator.evaluate groups fields by replica, including n=1.
    # This tester audits one prediction per sample in each independent round.
    for key in ("correct", "predictions", "references"):
        if key not in detail:
            raise Failure(f"AISBench 逐题评分缺少 {key}", "upstream_result_invalid", 1)
        if isinstance(detail[key], list):
            if len(detail[key]) != 1:
                raise Failure(f"AISBench 逐题评分 {key} 必须只有一次推理结果", "upstream_result_invalid", 1)
            detail[key] = detail[key][0]
    if type(detail["correct"]) is not bool or isinstance(detail["predictions"], (list, dict)) or isinstance(detail["references"], (list, dict)):
        raise Failure("AISBench 逐题评分格式已变化", "upstream_result_invalid", 1)
    return detail


def map_accuracy(directory, rows, labels):
    records = audit_records(directory, len(rows))
    jsonl(directory / "requests.jsonl", records)
    successes = [r for r in records if r.get("success") is True]
    complete = len(successes) == len(rows)
    labels = dict(labels, complete=complete)
    counts = {"selected": len(rows), "request_success": len(successes), "request_failed": len(rows) - len(successes)}
    metrics = [metric("samples", value, "count", "count", dict(labels, kind=key)) for key, value in counts.items()]
    if not complete:
        # Do not turn AISBench's synthesized predictions for failed requests into a full score.
        metrics.append(metric("accuracy", None, "percent", "mean", labels, notes="请求未完整执行，成绩不完整"))
        return metrics, Failure("有请求失败或缺失；保留原始结果，不报告完整成绩", "incomplete_requests", 1)
    result_files = list(directory.glob(f"aisbench/**/results/{MODEL}/{DATA}.json"))
    if len(result_files) != 1:
        raise Failure("没有唯一的 AISBench 评分文件", "upstream_result_invalid", 1)
    result = json.loads(result_files[0].read_text(encoding="utf-8"))
    details = result.get("details", {})
    if not isinstance(details, dict) or set(details) != {str(i) for i in range(len(rows))}:
        raise Failure("AISBench 逐题评分缺失或 ID 不一致", "upstream_result_invalid", 1)
    scored = []
    for index, record in enumerate(records):
        detail = single_score_detail(details[str(index)])
        expected_answer = rows[index]["answer"].split("#### ")[1].replace(",", "")
        if detail.get("origin_prediction") != record["prediction"] or str(detail["references"]) != expected_answer:
            raise Failure("评分与实际回答或参考答案不一致", "upstream_result_invalid", 1)
        scored.append(dict(id=index, question=rows[index]["question"], **detail))
    correct = sum(r["correct"] for r in scored)
    accuracy = 100 * correct / len(rows)
    upstream = result.get("accuracy")
    if type(upstream) not in (float, int) or not math.isfinite(upstream) or not math.isclose(accuracy, upstream, abs_tol=0.0001):
        raise Failure("AISBench 汇总分数与逐题分数不一致", "upstream_result_invalid", 1)
    jsonl(directory / "sample_scores.jsonl", scored)
    counts = {"scored": len(scored), "correct": correct, "extraction_failed": sum(r["predictions"] == "NULL" for r in scored)}
    metrics.extend(metric("samples", value, "count", "count", dict(labels, kind=key)) for key, value in counts.items())
    metrics.append(metric("accuracy", accuracy, "percent", "mean", labels, len(rows)))
    return metrics, None
