"""Prefix + three random tokens + suffix, following aisbench_auto_tools_prefix.

No process-global picked_ids file; all random choices belong to this invocation.
"""
import random
from pathlib import Path

from runtime import Failure, digest, input_path, jsonl, read_jsonl, select_rows


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def fixed_text(tokenizer, text, length):
    if length == 0:
        return ""
    tokens = encode(tokenizer, text)
    if not tokens:
        raise Failure("语料无法编码成 token")
    tokens = (tokens * (length // len(tokens) + 2))[:length]
    for _ in range(12):
        candidate = decode(tokenizer, tokens)
        actual = encode(tokenizer, candidate)
        if len(actual) == length:
            return candidate
        if not actual:
            break
        tokens = (actual + encode(tokenizer, text))[:length]
    raise Failure("token 转文本后无法满足指定长度，请检查 tokenizer")


def common_prefix(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def chat_tokens(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def validate_prefixes(tokenizer, rows, warmup):
    prefixes = list(dict.fromkeys(row["question"] for row in warmup))
    evidence = []
    prefix_tokens = [encode(tokenizer, text) for text in prefixes]
    prefix_chats = [chat_tokens(tokenizer, text) for text in prefixes]
    for index, row in enumerate(rows):
        raw_tokens, actual = encode(tokenizer, row["question"]), chat_tokens(tokenizer, row["question"])
        candidates = []
        for prefix_id, (text, tokens, chat) in enumerate(zip(prefixes, prefix_tokens, prefix_chats)):
            if not tokens or not row["question"].startswith(text) or raw_tokens[:len(tokens)] != tokens:
                continue
            reused = common_prefix(chat, actual)
            if reused >= len(tokens):
                candidates.append((prefix_id, reused))
        if not candidates:
            raise Failure(f"第 {index} 条数据与预热文件没有匹配的 token 前缀（含聊天模板检查）")
        prefix_id, reused = max(candidates, key=lambda item: item[1])
        evidence.append(dict(id=index, prefix_id=prefix_id, shared_chat_tokens=reused, chat_tokens=len(actual)))
    used = {item["prefix_id"] for item in evidence}
    if len(used) != len(prefixes):
        raise Failure("预热文件含正式数据未使用的前缀，请使用对应的预热文件")
    return evidence, len(prefixes)


def generate(tokenizer, args, corpus):
    rng = random.Random(args.seed)
    texts = [row["question"] for row in read_jsonl(corpus)]
    prefix_len = int(args.input_len * args.repeat_rate) if args.dataset_type == "prefix_cache" else 0
    if 0 < prefix_len < args.input_len and args.input_len - prefix_len < 4:
        raise Failure("前缀之后必须至少留 3 个分隔 token 和 1 个后缀 token")
    prefixes = []
    for _ in range(args.prefix_num if prefix_len else 0):
        for attempt in range(64):
            value = fixed_text(tokenizer, rng.choice(texts), prefix_len)
            if value not in prefixes:
                prefixes.append(value)
                break
        else:
            raise Failure("指定长度无法生成足够的不同前缀")
    special = set(tokenizer.all_special_ids)
    vocabulary = [i for i in sorted(set(tokenizer.get_vocab().values())) if i not in special]
    if len(vocabulary) < 3:
        raise Failure("tokenizer 的普通 token 不足 3 个")
    rows = []
    separators = set()
    for index in range(args.data_num):
        prefix = prefixes[index % len(prefixes)] if prefixes else ""
        if prefix_len == args.input_len:
            text = prefix
        elif prefix_len == 0:
            text = fixed_text(tokenizer, rng.choice(texts), args.input_len)
        else:
            for attempt in range(128):
                ids = tuple(rng.sample(vocabulary, 3))
                separator = decode(tokenizer, ids)
                if ids in separators or len(encode(tokenizer, separator)) != 3 or "\ufffd" in separator:
                    continue
                suffix = fixed_text(tokenizer, rng.choice(texts), args.input_len - prefix_len - 3 + 16)
                text = fixed_text(tokenizer, prefix + separator + suffix, args.input_len)
                prefix_ids = encode(tokenizer, prefix)
                if text.startswith(prefix) and encode(tokenizer, text)[:len(prefix_ids)] == prefix_ids:
                    separators.add(ids)
                    break
            else:
                raise Failure("拼接后无法保持指定长度和公共 token 前缀")
        if len(encode(tokenizer, text)) != args.input_len:
            raise Failure("生成数据长度不一致")
        rows.append(dict(question=text, answer="none"))
    warmup = [dict(question=prefix, answer="none") for prefix in prefixes for _ in range(args.dp)]
    return rows, warmup


def prepare_data(args, tokenizer, output):
    evidence = {}
    if args.dataset:
        source = input_path(args.dataset)
        rows = select_rows(read_jsonl(source), args.data_num)
        evidence["input_sha256"] = digest(source)
        warmup = read_jsonl(input_path(args.warmup_dataset)) if args.warmup_dataset else []
        if args.warmup_dataset:
            evidence["warmup_input_sha256"] = digest(args.warmup_dataset)
    else:
        corpus = Path(__file__).parent / "data/GSM8K.jsonl"
        rows, warmup = generate(tokenizer, args, corpus)
        evidence["corpus_sha256"] = digest(corpus)
    if warmup:
        pairs, prefix_count = validate_prefixes(tokenizer, rows, warmup)
        evidence["prefix_correspondence"] = pairs
    else:
        prefix_count = 0
    for name, data in (("measurement", rows), ("warmup", warmup)):
        if data:
            jsonl(output / "raw/datasets" / name / "test.jsonl", [dict(question=row["question"], answer="none") for row in data])
            jsonl(output / "raw/datasets" / name / "train.jsonl", [dict(question=data[0]["question"], answer="none")])
            evidence[name + "_sha256"] = digest(output / "raw/datasets" / name / "test.jsonl")
    evidence["token_lengths"] = [dict(id=index, content=len(encode(tokenizer, row["question"])), chat=len(chat_tokens(tokenizer, row["question"]))) for index, row in enumerate(rows)]
    evidence["generator"] = "prefix-separator3-suffix-v1"
    return rows, warmup, prefix_count, evidence
