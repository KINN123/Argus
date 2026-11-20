"""Snapshot differences, retaining source labels and independent cache groups."""
import json
import math
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler
import os

from prometheus_client.parser import text_string_to_metric_families

from runtime import Failure, dump, metric, now, redact

GROUPS = {
    "ucm_connector": {
        "hbm": ("ucm:total_prefix_query_tokens_total", "ucm:gpu_hbm_hit_tokens_total", "connector_queries"),
        "ucm": ("ucm:total_prefix_query_tokens_total", "ucm:ucm_hit_tokens_total", "connector_queries"),
    },
    "vllm_prefix": {
        "hbm": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "hbm_queries"),
        "external": ("external_prefix_cache_queries_total", "external_prefix_cache_hits_total", "external_queries"),
    },
    "vllm_prompt": {
        "hbm": ("vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total", "prompt_tokens"),
        "external": ("vllm:prompt_tokens_total", "external_prefix_cache_hits_total", "prompt_tokens"),
    },
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Never send a metrics credential to a redirected destination.
        return None


def parse_snapshot(text, model):
    samples, starts = {}, {}
    relevant = {name for groups in GROUPS.values() for values in groups.values() for name in values[:2]}
    relevant.update({"vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total"})
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in relevant and sample.name != "process_start_time_seconds":
                continue
            if any(sample.labels.get(key, model) != model for key in ("model_name", "model")):
                continue
            if not math.isfinite(sample.value) or sample.value < 0:
                raise ValueError(f"无效计数：{sample.name}")
            labels = tuple(sorted(sample.labels.items()))
            key = (sample.name, labels)
            if key in samples:
                raise ValueError("重复的 Prometheus 序列")
            samples[key] = sample.value
            if sample.name == "process_start_time_seconds":
                starts[labels] = sample.value
    return {"samples": samples, "starts": starts}


def capture(endpoints, model, timeout, directory, label):
    snapshots, evidence = {}, {}
    for index, endpoint in enumerate(endpoints):
        attempts = []
        for attempt in range(2):
            started = now()
            try:
                headers = {"Accept": "text/plain"}
                if os.environ.get("METRICS_API_KEY"):
                    headers["Authorization"] = "Bearer " + os.environ["METRICS_API_KEY"]
                with build_opener(NoRedirect()).open(Request(endpoint, headers=headers), timeout=timeout) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("指标响应超过 16 MiB")
                content = raw.decode("utf-8")
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{label}-{index}.prom").write_text(redact(content), encoding="utf-8")
                snapshots[endpoint] = parse_snapshot(content, model)
                attempts.append(dict(started_at=started, finished_at=now(), success=True))
                break
            except (OSError, ValueError) as exc:
                attempts.append(dict(started_at=started, finished_at=now(), success=False, error=redact(str(exc))))
                snapshots[endpoint] = {"error": redact(str(exc))}
        evidence[endpoint] = attempts
    dump(directory / f"{label}-collection.json", evidence)
    return snapshots


def group_series(snapshot, query, hit):
    if "error" in snapshot:
        raise ValueError(snapshot["error"])
    samples = snapshot["samples"]
    # Some engines use the vllm: alias. Never add both representations together.
    pairs = [(query, hit)]
    if hit.startswith("external_"):
        pairs.append(("vllm:" + query if query.startswith("external_") else query, "vllm:" + hit))
    # A shared denominator cannot establish that an external hit counter exists.
    present = [pair for pair in pairs if any(key[0] == pair[1] for key in samples)]
    if not present:
        raise ValueError(f"缺少命中计数 {hit}（或其 vllm: 别名）；查询计数应为 {query}")
    if len(present) > 1:
        raise ValueError(f"同时出现两套命中计数别名：{hit}")
    query, hit = present[0]
    queries = {labels: value for (name, labels), value in samples.items() if name == query}
    hits = {labels: value for (name, labels), value in samples.items() if name == hit}
    if not queries or set(queries) != set(hits):
        raise ValueError("查询和命中计数的标签不匹配或缺失")
    if any(hits[key] > queries[key] for key in queries):
        raise ValueError("累计命中大于累计查询")
    return queries, hits, (query, hit)


def compare(before, after, profile, labels):
    metrics, summary, complete = [], {}, True
    endpoints = sorted(set(before) | set(after))
    for cache, (query, hit, denominator) in GROUPS[profile].items():
        valid, errors, identities = [], {}, set()
        for endpoint in endpoints:
            try:
                first, last = before[endpoint], after[endpoint]
                q0, h0, names0 = group_series(first, query, hit)
                q1, h1, names1 = group_series(last, query, hit)
                if first["starts"] != last["starts"]:
                    raise ValueError("进程重启或进程标识集合变化")
                if set(q0) != set(q1) or names0 != names1:
                    raise ValueError("实例/标签/计数名称发生变化")
                local = []
                for identity in q0:
                    source_labels = dict(identity)
                    if any(key in source_labels for key in ("instance", "pod", "pod_name")):
                        if identity in identities:
                            raise ValueError("不同地址重复暴露同一实例的计数")
                        identities.add(identity)
                    queries, hits = q1[identity] - q0[identity], h1[identity] - h0[identity]
                    if queries < 0 or hits < 0:
                        raise ValueError("计数发生重置")
                    if hits > queries:
                        raise ValueError("命中增量大于查询增量")
                    local.append(dict(endpoint=endpoint, labels=source_labels, queries=queries, hits=hits,
                                      rate=100 * hits / queries if queries else None,
                                      counter_names=names1, process_identity_available=bool(first["starts"])))
                valid.extend(local)
            except (KeyError, ValueError) as exc:
                errors[endpoint] = str(exc)
        group_complete = not errors and bool(valid)
        complete = complete and group_complete
        total_queries = sum(item["queries"] for item in valid) if group_complete else None
        total_hits = sum(item["hits"] for item in valid) if group_complete else None
        total_rate = 100 * total_hits / total_queries if total_queries else None
        summary[cache] = dict(complete=group_complete, queries=total_queries, hits=total_hits, rate=total_rate,
                              instances=valid, errors=errors, denominator=denominator)
        scopes = [(dict(scope="instance", instance=item["endpoint"], source_labels=json.dumps(item["labels"], sort_keys=True)), item["queries"], item["hits"], item["rate"]) for item in valid]
        scopes.append((dict(scope="all"), total_queries, total_hits, total_rate))
        for scope, queries, hits, rate in scopes:
            dimensions = dict(labels, cache=cache, denominator=denominator, **scope)
            note = "采集不完整：" + json.dumps(errors, ensure_ascii=False) if queries is None else "本窗口无新增查询" if queries == 0 else None
            for name, value, unit, statistic in (("prefix_cache_queries", queries, "tokens", "sum"),
                                                ("prefix_cache_hits", hits, "tokens", "sum"),
                                                ("prefix_cache_hit_rate", rate, "percent", "mean")):
                metrics.append(metric(name, value, unit, statistic, dimensions, notes=note))
    return metrics, summary, complete


def require_valid(snapshot, args, labels):
    _, summary, complete = compare(snapshot, snapshot, choose_profile(snapshot, args.metrics_profile), labels)
    if not complete and args.metrics_policy == "required":
        raise Failure("正式请求前缓存计数不完整：" + json.dumps(summary, ensure_ascii=False), "metrics_unavailable", 1)


def choose_profile(snapshot, requested):
    if requested != "auto":
        return requested
    for profile in ("ucm_connector", "vllm_prefix", "vllm_prompt"):
        _, _, complete = compare(snapshot, snapshot, profile, {})
        if complete:
            return profile
    for profile in ("vllm_prefix", "vllm_prompt", "ucm_connector"):
        _, summary, _ = compare(snapshot, snapshot, profile, {})
        if summary["hbm"]["complete"]:
            return profile
    return "vllm_prefix"


def settle(seconds):
    if seconds:
        time.sleep(seconds)
