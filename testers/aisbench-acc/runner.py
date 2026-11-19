"""GSM8K accuracy tester; service deployment and UCM switches stay outside."""
import hashlib
import re

from runtime import (Failure, Parser, dump, input_path, integer, jsonl, number,
                     official_template, public_args, read_jsonl, run_main,
                     select_rows, tokenizer_info, verify_aisbench, digest)
from result_mapper import map_accuracy


def parser():
    p = Parser(description="AISBench GSM8K 精度测试", allow_abbrev=False)
    public_args(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset_config", choices=("gsm8k_4shot_cot", "gsm8k_jsonl_0shot"), default="gsm8k_4shot_cot")
    p.add_argument("--max_samples", type=integer(0), default=0)
    p.add_argument("--output_len", type=integer(), default=1024)
    p.add_argument("--temperature", type=number(0, 2), default=0)
    p.add_argument("--top_p", type=number(0, 1, open_min=True), default=1)
    p.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    p.add_argument("--replay_rounds", type=int, choices=(1, 2), default=1)
    return p


def prepare_data(args, output):
    four_shot = args.dataset_config == "gsm8k_4shot_cot"
    path = input_path(args.dataset, directory=four_shot)
    test_path = path / "test.jsonl" if four_shot else path
    train = read_jsonl(path / "train.jsonl") if four_shot else []
    rows = select_rows(read_jsonl(test_path), args.max_samples)
    for row in rows + train:
        answer = row.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
            raise Failure("每行需要有效的 GSM8K answer")
        answer = str(answer).strip()
        if "#### " not in answer:
            if four_shot:
                raise Failure("标准 GSM8K 的答案必须包含 #### 后的数字")
            answer = "#### " + answer
        final = answer.split("#### ")
        if len(final) != 2 or not re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", final[1]):
            raise Failure("GSM8K answer 必须有明确的数字答案，不能是 none")
        row["answer"] = answer
    data = output / "raw/dataset"
    jsonl(data / "test.jsonl", rows)
    # The official four examples are fixed in the prompt, not sampled from train.
    jsonl(data / "train.jsonl", train or [rows[0]])
    evidence = dict(input_sha256=digest(test_path), selected_sha256=digest(data / "test.jsonl"), sample_ids=list(range(len(rows))))
    if four_shot:
        evidence["train_sha256"] = digest(path / "train.jsonl")
    return data, rows, evidence


def work(execution, args):
    data, rows, evidence = prepare_data(args, execution.output)
    root, versions = verify_aisbench(execution.log)
    execution.result["metadata"].update(versions)
    tokenizer, tokenizer_files = tokenizer_info(input_path(args.model_path, directory=True))
    if args.thinking != "default" and "enable_thinking" not in str(tokenizer.chat_template):
        raise Failure("本地 chat_template 不支持 enable_thinking", "unsupported_backend")
    template = official_template(root, args.dataset_config)
    execution.result["workload"] = dict(model_name=args.model_name, dataset=args.dataset_config, concurrency=args.concurrency,
                                         output_len=args.output_len)
    metadata = dict(versions, **evidence, tokenizer_files=tokenizer_files,
                    template_sha256=hashlib.sha256(template.encode()).hexdigest(), seed_support="requested_unverified",
                    thinking_support="default" if args.thinking == "default" else "template_supported_server_unverified")
    execution.result["metadata"].update(metadata)
    dump(execution.output / "raw/resolved.json", dict(metadata, parameters={k: v for k, v in vars(args).items() if k != "explicit"}))
    for round_id in range(1, args.replay_rounds + 1):
        directory = execution.output / f"raw/round-{round_id}"
        code = execution.run_aisbench(directory, args, data, template, perf=False)
        labels = dict(dataset="gsm8k", preset=args.dataset_config, subset="test", round=round_id, phase="measurement")
        try:
            metrics, error = map_accuracy(directory, rows, labels)
            execution.result["metrics"].extend(metrics)
        except Failure:
            if code:
                raise Failure(f"AISBench 退出码 {code}，日志见本轮 aisbench.log", "benchmark_failed", 1)
            raise
        if code or error:
            raise error or Failure(f"AISBench 退出码 {code}", "benchmark_failed", 1)
    execution.result["notes"] = "分数表示答题结果；执行成功不代表 UCM 已实际加载缓存。"


if __name__ == "__main__":
    raise SystemExit(run_main("aisbench-acc", parser, work))
