"""Prefix + three random tokens + suffix, following aisbench_auto_tools_prefix.

No process-global picked_ids file; all random choices belong to this invocation.
"""
import random
from pathlib import Path

from runtime import Failure, chat_token_ids, digest, input_path, jsonl, read_jsonl, select_rows


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
    return chat_token_ids(tokenizer, [{"role": "user", "content": text}])


def matching_prefix(text, raw_tokens, actual, prefix, prefix_tokens, prefix_chat):
    if not prefix_tokens or not text.startswith(prefix) or raw_tokens[:len(prefix_tokens)] != prefix_tokens:
        return None
    reused = common_prefix(prefix_chat, actual)
    return reused if reused >= len(prefix_tokens) else None


def validate_prefixes(tokenizer, rows, warmup):
    prefixes = list(dict.fromkeys(row["question"] for row in warmup))
    evidence = []
    prefix_tokens = [encode(tokenizer, text) for text in prefixes]
    prefix_chats = [chat_tokens(tokenizer, text) for text in prefixes]
    buckets = {}
    for i, tokens in enumerate(prefix_tokens):
        if tokens:
            buckets.setdefault(tokens[0], []).append(i)
    for index, row in enumerate(rows):
        raw_tokens, actual = encode(tokenizer, row["question"]), chat_tokens(tokenizer, row["question"])
        candidates = []
        for prefix_id in buckets.get(raw_tokens[0] if raw_tokens else None, []):
            text, tokens, chat = prefixes[prefix_id], prefix_tokens[prefix_id], prefix_chats[prefix_id]
            reused = matching_prefix(row["question"], raw_tokens, actual, text, tokens, chat)
            if reused is not None:
                candidates.append((prefix_id, reused))
        if not candidates:
            text_matches = [i for i, text in enumerate(prefixes) if row["question"].startswith(text)]
            raw_matches = [i for i in text_matches if prefix_tokens[i] and raw_tokens[:len(prefix_tokens[i])] == prefix_tokens[i]]
            reason = "文本没有共同前缀" if not text_matches else "裸 token 前缀不一致" if not raw_matches else "; ".join(
                f"前缀 {i}: 共享聊天 token={common_prefix(prefix_chats[i], actual)}, 要求至少 {len(prefix_tokens[i])}" for i in raw_matches[:3])
            raise Failure(f"第 {index} 条数据与预热文件没有匹配的 token 前缀：{reason}；数据见 raw/datasets")
        prefix_id, reused = max(candidates, key=lambda item: item[1])
        evidence.append(dict(id=index, prefix_id=prefix_id, shared_chat_tokens=reused, chat_tokens=len(actual)))
    used = {item["prefix_id"] for item in evidence}
    if len(used) != len(prefixes):
        raise Failure("预热文件含正式数据未使用的前缀，请使用对应的预热文件")
    return evidence, len(prefixes)


def generate_corpus(tokenizer, args, corpus):
    args.effective_seed = args.seed or random.SystemRandom().randrange(1, 2**63)
    rng = random.Random(args.effective_seed)
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
    prefix_encodings = {prefix: (encode(tokenizer, prefix), chat_tokens(tokenizer, prefix)) for prefix in prefixes}
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
                prefix_ids, prefix_chat = prefix_encodings[prefix]
                if matching_prefix(text, encode(tokenizer, text), chat_tokens(tokenizer, text), prefix, prefix_ids, prefix_chat) is not None:
                    separators.add(ids)
                    break
            else:
                raise Failure("拼接后无法保持指定长度、裸 token 前缀和聊天 token 前缀")
        if len(encode(tokenizer, text)) != args.input_len:
            raise Failure("生成数据长度不一致")
        rows.append(dict(question=text, answer="none"))
    warmup = [dict(question=prefix, answer="none") for prefix in prefixes for _ in range(args.dp)]
    return rows, warmup


def random_text(tokenizer, rng, vocabulary, length):
    if not length:
        return ""
    ids = []
    for _ in range(128):
        need = max(1, length - len(ids))
        ids.extend(rng.choices(vocabulary, k=need + 8))
        text = decode(tokenizer, ids[:length])
        ids = encode(tokenizer, text)
        if len(ids) == length and "\ufffd" not in text and text.strip():
            return text
        if len(ids) >= length:
            ids = ids[:max(0, length - 8)]
    raise Failure("随机 token 转文本后无法达到目标长度")


def generate(tokenizer, args, corpus):
    if getattr(args, "data_source", "random") == "corpus":
        return generate_corpus(tokenizer, args, corpus)
    args.effective_seed = args.seed or random.SystemRandom().randrange(1, 2**63)
    rng = random.Random(args.effective_seed)
    special = set(tokenizer.all_special_ids)
    vocabulary = []
    for token in sorted(set(tokenizer.get_vocab().values()) - special):
        text = decode(tokenizer, [token])
        if text.strip() and "\ufffd" not in text and encode(tokenizer, text) == [token]:
            vocabulary.append(token)
    if len(vocabulary) < 3:
        raise Failure("tokenizer 缺少可往返编码的普通 token")
    lengths = []
    for _ in range(args.data_num):
        if args.length_mean is not None:
            size = round(rng.gauss(args.length_mean, args.length_std))
            size = max(args.length_min or 4, size)
            if args.length_max is not None:
                size = min(size, args.length_max)
        elif args.length_min is not None:
            size = rng.randint(args.length_min, args.length_max)
        else:
            size = args.input_len
        lengths.append(size)
    rows, warmup, pool = [], [], {}
    used_prefixes, used_separators = set(), set()
    for index, length in enumerate(lengths):
        prefix_len = int(length * args.repeat_rate) if args.dataset_type == "prefix_cache" else 0
        private = args.prefix_mode == "private"
        gap = 1 if private else 3
        if 0 < prefix_len < length and length - prefix_len <= gap:
            raise Failure("前缀后剩余长度不足以容纳分隔符和后缀")
        key = (index if private else index % args.prefix_num, prefix_len)
        if prefix_len and key not in pool:
            for _ in range(128):
                prefix = random_text(tokenizer, rng, vocabulary, prefix_len)
                if prefix not in used_prefixes:
                    break
            else:
                raise Failure("无法生成足够的不同随机前缀")
            pool[key] = prefix
            used_prefixes.add(prefix)
            warmup.extend(dict(question=prefix, answer="none") for _ in range(args.dp))
        prefix = pool.get(key, "")
        if prefix_len == length:
            text = prefix
        elif not prefix_len:
            text = random_text(tokenizer, rng, vocabulary, length)
        else:
            prefix_ids, prefix_chat = encode(tokenizer, prefix), chat_tokens(tokenizer, prefix)
            for _ in range(128):
                separator = " " if private else random_text(tokenizer, rng, vocabulary, 3)
                if not private and separator in used_separators:
                    continue
                suffix = random_text(tokenizer, rng, vocabulary, length - prefix_len + 8)
                text = fixed_text(tokenizer, prefix + separator + suffix, length)
                if matching_prefix(text, encode(tokenizer, text), chat_tokens(tokenizer, text), prefix, prefix_ids, prefix_chat) is not None:
                    used_separators.add(separator)
                    break
            else:
                raise Failure("随机数据不能保持指定 token 长度和聊天前缀")
        rows.append(dict(question=text, answer="none"))
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
        evidence["effective_seed"] = getattr(args, "effective_seed", args.seed)
        if args.data_source == "corpus":
            evidence["corpus_sha256"] = digest(corpus)
    for name, data in (("measurement", rows), ("warmup", warmup)):
        if data:
            jsonl(output / "raw/datasets" / name / "test.jsonl", [dict(question=row["question"], answer="none") for row in data])
            jsonl(output / "raw/datasets" / name / "train.jsonl", [dict(question=data[0]["question"], answer="none")])
            evidence[name + "_sha256"] = digest(output / "raw/datasets" / name / "test.jsonl")
    if warmup:
        pairs, prefix_count = validate_prefixes(tokenizer, rows, warmup)
        evidence["prefix_correspondence"] = pairs
    else:
        prefix_count = 0
    evidence["token_lengths"] = [dict(id=index, content=len(encode(tokenizer, row["question"])), chat=len(chat_tokens(tokenizer, row["question"]))) for index, row in enumerate(rows)]
    evidence["generator"] = args.data_source + "-" + args.prefix_mode + "-v3"
    return rows, warmup, prefix_count, evidence
