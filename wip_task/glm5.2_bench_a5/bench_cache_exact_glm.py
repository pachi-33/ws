#!/usr/bin/env python3
"""Exact token-ID cache workload with per-DP warmup and native SGLang metrics.

This launcher changes only this benchmark process's dataset/request adapters.
It does not change the installed SGLang package or the server's load balancing.
"""

import argparse
import contextvars
import json
import os
import random
import sys
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


class CacheValidationError(RuntimeError):
    pass


def server_info_object(info):
    if "decode" in info:
        info = info["decode"][0]
    return info


def request_json(url, headers, payload=None, timeout=3600, decode_json=True):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url, data=body, headers={**headers, "Content-Type": "application/json"}
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response) if decode_json else response.read().decode()


def build_workload(tokenizer, input_len, prefix_len, count, page_size, seed):
    """Different markers within the first suffix page prevent extra page hits."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab = sorted(
        {
            v
            for v in tokenizer.get_vocab().values()
            if isinstance(v, int) and v >= 0 and v not in special
        }
    )
    if len(vocab) < 4:
        raise ValueError("Need at least four non-special vocabulary tokens")
    guard_a, guard_b = vocab[:2]
    marker_vocab = vocab[2:]
    base, width, capacity = len(marker_vocab), 1, len(marker_vocab)
    while capacity < count:
        width += 1
        capacity *= base
    suffix_len = input_len - prefix_len
    if width > min(page_size, suffix_len):
        raise ValueError(
            "Too many requests to give every suffix a distinct first KV page"
        )
    rng = random.Random(seed)
    prefix = rng.choices(vocab, k=prefix_len)
    prompts = []
    for i in range(count):
        marker, index = [marker_vocab[0]] * width, i
        for pos in range(width - 1, -1, -1):
            marker[pos] = marker_vocab[index % base]
            index //= base
        suffix = marker + rng.choices(vocab, k=suffix_len - width)
        prompts.append(prefix + suffix)
    # Guards have different first tokens from all measured suffixes.
    guard_len = min(page_size, suffix_len)
    return prefix, prompts, [guard_a] * guard_len, [guard_b] * guard_len


def dp_rank(meta, dp_size):
    rank = meta.get("dp_rank")
    if rank is None and dp_size == 1:
        return 0
    return rank


def warm_all_dp(
    base_url, headers, prefix, guard_a, guard_b, dp_size, timeout, strict=False
):
    """Warm beyond the prefix boundary, then probe with a different guard.

    The probe cannot match the warmup's guard and cannot warm a measured suffix.
    Both calls are outside the measured workload.
    """
    if not prefix:
        return []
    reports = []
    for rank in range(dp_size):

        def call(ids):
            payload = {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "stream": False,
            }
            if dp_size > 1:
                payload["routed_dp_rank"] = rank
            result = request_json(base_url + "/generate", headers, payload, timeout)
            if isinstance(result, list):
                if len(result) != 1:
                    raise RuntimeError("Unexpected batched warmup response")
                result = result[0]
            meta = result.get("meta_info") or {}
            reason = meta.get("finish_reason") or {}
            if isinstance(reason, dict) and reason.get("type") == "abort":
                raise RuntimeError(f"DP{rank} warmup aborted: {reason}")
            if meta.get("prompt_tokens") != len(ids) or dp_rank(meta, dp_size) != rank:
                raise RuntimeError(
                    f"DP{rank} warmup routing/token count not confirmed: {meta}"
                )
            return meta

        call(prefix + guard_a)
        meta = call(prefix + guard_b)
        cached = meta.get("cached_tokens")
        if (
            not isinstance(cached, int)
            or isinstance(cached, bool)
            or not 0 <= cached <= len(prefix) + len(guard_b)
        ):
            raise RuntimeError(
                f"DP{rank} cache probe returned invalid cached_tokens={cached}"
            )
        matched = cached == len(prefix)
        if not matched and strict:
            raise RuntimeError(
                f"DP{rank} cache probe failed: expected {len(prefix)} cached tokens, got {cached}"
            )
        reports.append(
            {
                "dp_rank": rank,
                "cached_tokens": cached,
                "expected_cached_tokens": len(prefix),
                "routing_verified": True,
                "verified": matched,
            }
        )
        if matched:
            print(
                f"DP{rank}: shared prefix verified, {cached} cached tokens", flush=True
            )
        else:
            print(
                f"WARNING DP{rank}: cache probe got {cached} cached tokens, target {len(prefix)}; continuing in report mode",
                flush=True,
            )
    return reports


def validate_records(
    records, input_len, output_len, prefix_len, dp_size, tolerance, strict=False
):
    failures = []
    deviations = []
    matched_count = 0
    per_dp = {
        rank: {
            "dp_rank": rank,
            "requests": 0,
            "passed_requests": 0,
            "cache_target_matched_requests": 0,
        }
        for rank in range(dp_size)
    }
    for i, record in enumerate(records):
        meta = record.get("meta_info") or {}
        rank = dp_rank(meta, dp_size)
        cached = meta.get("cached_tokens")
        errors = []
        reason = meta.get("finish_reason") or {}
        if isinstance(reason, dict) and reason.get("type") == "abort":
            errors.append(f"request aborted: {reason}")
        if not record.get("success"):
            errors.append(record.get("error") or "request failed")
        if meta.get("prompt_tokens") != input_len:
            errors.append(
                f"prompt_tokens={meta.get('prompt_tokens')}, expected {input_len}"
            )
        if meta.get("completion_tokens") != output_len:
            errors.append(
                f"completion_tokens={meta.get('completion_tokens')}, expected {output_len}"
            )
        if record.get("ttft", 0) <= 0:
            errors.append("first nonempty output was not observed")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank not in per_dp:
            errors.append(f"missing or invalid dp_rank={rank}")
        cached_valid = (
            isinstance(cached, int)
            and not isinstance(cached, bool)
            and 0 <= cached <= input_len
        )
        matched = cached_valid and abs(cached - prefix_len) <= tolerance
        matched_count += int(matched)
        if not cached_valid:
            errors.append(f"missing or invalid cached_tokens={cached}")
        elif not matched:
            deviation = {
                "request_index": i,
                "dp_rank": rank,
                "cached_tokens": cached,
                "actual_hit_rate_pct": 100 * cached / input_len,
            }
            deviations.append(deviation)
            if strict:
                errors.append(
                    f"cached_tokens={cached}, expected {prefix_len} +/- {tolerance}"
                )
        if errors:
            failures.append({"request_index": i, "dp_rank": rank, "reasons": errors})
        if isinstance(rank, int) and rank in per_dp:
            per_dp[rank]["requests"] += 1
            per_dp[rank]["passed_requests"] += int(not errors)
            per_dp[rank]["cache_target_matched_requests"] += int(matched)
    return {
        "valid": not failures,
        "mode": "strict" if strict else "report",
        "cache_target_matched": matched_count == len(records),
        "deviating_requests": len(deviations),
        "cache_deviations": deviations,
        "checked_requests": len(records),
        "expected_cached_tokens": prefix_len,
        "expected_hit_rate_pct": 100 * prefix_len / input_len,
        "tolerance_tokens": tolerance,
        "failed_requests": len(failures),
        "failures": failures,
        "per_dp": list(per_dp.values()),
    }


def replace_last_jsonl(path, result):
    """Preserve previous runs when an output filename is reused."""
    with Path(path).open("r+", encoding="utf-8") as handle:
        last_offset = None
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                last_offset = offset
        if last_offset is None:
            raise RuntimeError("Native benchmark did not write a result")
        handle.seek(last_offset)
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.truncate()


def install_adapter(serving, exact):
    from sglang.benchmark.datasets.common import DatasetRow

    original_run = serving.run_benchmark
    original_request = serving.async_request_sglang_generate
    original_json = serving.orjson
    current_record = contextvars.ContextVar("cache_bench_record", default=None)
    state = {}

    class MetadataJSON:
        def __getattr__(self, name):
            return getattr(original_json, name)

        def loads(self, *args, **kwargs):
            data = original_json.loads(*args, **kwargs)
            record = current_record.get()
            if record is not None and isinstance(data, dict):
                meta = data.get("meta_info")
                if isinstance(meta, dict):
                    record.setdefault("meta_info", {}).update(meta)
            return data

    async def measured_request(request_func_input, pbar=None):
        record = {}
        token = current_record.set(record)
        try:
            output = await original_request(
                request_func_input=request_func_input, pbar=pbar
            )
        finally:
            current_record.reset(token)
        meta = record.get("meta_info") or {}
        # Use server token counts; native latency/TTFT/ITL remain unchanged.
        if isinstance(meta.get("prompt_tokens"), int):
            output.prompt_len = meta["prompt_tokens"]
        if isinstance(meta.get("completion_tokens"), int):
            output.output_len = meta["completion_tokens"]
        if isinstance(meta.get("cached_tokens"), int):
            output.cached_tokens = meta["cached_tokens"]
        if meta.get("cached_tokens_details") is not None:
            output.cached_tokens_details = meta["cached_tokens_details"]
        record.update(
            success=output.success,
            error=output.error,
            latency=output.latency,
            ttft=output.ttft,
        )
        rid = request_func_input.extra_request_body["rid"]
        state["records"][rid] = record
        return output

    def exact_dataset(args, tokenizer, model_id=None):
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        base_url = base_url.rstrip("/")
        headers = serving.get_auth_headers()
        info = server_info_object(request_json(base_url + "/server_info", headers))
        page = int(info.get("page_size") or 0)
        dp_size = int(info.get("dp_size") or 0)
        if page <= 0 or dp_size <= 0:
            raise ValueError("Server must report positive page_size and dp_size")
        if exact.page_size != "auto" and int(exact.page_size) != page:
            raise ValueError(
                f"Configured page_size={exact.page_size}, server page_size={page}"
            )
        if exact.dp_size != "auto" and int(exact.dp_size) != dp_size:
            raise ValueError(
                f"Configured dp_size={exact.dp_size}, server dp_size={dp_size}"
            )
        if info.get("disable_radix_cache"):
            raise ValueError("Server prefix cache is disabled")
        if (
            info.get("max_req_input_len")
            and exact.input_len > info["max_req_input_len"]
        ):
            raise ValueError("Input length exceeds server max_req_input_len")
        prefix_len = exact.input_len * exact.hit_percent // 100 // page * page
        if exact.hit_percent > 0 and prefix_len == 0:
            raise ValueError("Requested nonzero prefix is smaller than one KV page")
        prefix, prompts, guard_a, guard_b = build_workload(
            tokenizer, exact.input_len, prefix_len, exact.num_requests, page, args.seed
        )
        # This endpoint returns plain text, unlike /generate and /server_info.
        request_json(
            base_url + "/flush_cache?timeout=60", headers, {}, 65, decode_json=False
        )
        probes = warm_all_dp(
            base_url,
            headers,
            prefix,
            guard_a,
            guard_b,
            dp_size,
            exact.warmup_timeout,
            exact.strict_cache_hit,
        )
        run_id = uuid.uuid4().hex
        state.update(
            records={},
            ids=[f"cachebench-{run_id}-{i}" for i in range(exact.num_requests)],
            configuration={
                "input_tokens": exact.input_len,
                "output_tokens": args.gsp_output_len,
                "requested_hit_rate_pct": exact.hit_percent,
                "expected_hit_rate_pct": 100 * prefix_len / exact.input_len,
                "prefix_tokens": prefix_len,
                "suffix_tokens": exact.input_len - prefix_len,
                "page_size": page,
                "dp_size": dp_size,
                "warmup_probes": probes,
                "strict_cache_hit": exact.strict_cache_hit,
            },
        )
        print(
            f"Exact workload: input={exact.input_len}, prefix={prefix_len}, aligned hit={100 * prefix_len / exact.input_len:.8f}%, DP={dp_size}",
            flush=True,
        )
        return [
            DatasetRow(
                prompt=ids,
                prompt_len=len(ids),
                output_len=args.gsp_output_len,
                extra_request_body={"rid": rid},
            )
            for ids, rid in zip(prompts, state["ids"])
        ]

    def run(args):
        if args.backend != "sglang" or args.warmup_requests != 0 or args.flush_cache:
            raise ValueError(
                "Exact cache benchmark requires --backend sglang --warmup-requests 0 and no --flush-cache"
            )
        if not args.output_file or not args.output_details or not args.cache_report:
            raise ValueError(
                "Require --output-file, --output-details and --cache-report"
            )
        extra_body = (
            json.loads(args.extra_request_body) if args.extra_request_body else {}
        )
        forbidden = {
            "routed_dp_rank",
            "data_parallel_rank",
            "input_ids",
            "text",
            "sampling_params",
            "stream",
        } & extra_body.keys()
        if forbidden:
            raise ValueError(
                f"Do not override exact workload fields: {sorted(forbidden)}"
            )
        if args.disable_stream or args.disable_ignore_eos:
            raise ValueError(
                "Exact cache benchmark requires streaming and ignore_eos for fixed output lengths"
            )
        os.environ.pop("SGLANG_IS_IN_CI", None)
        result = original_run(args)
        records = [
            state["records"].get(rid, {"error": "request result missing"})
            for rid in state["ids"]
        ]
        config = state["configuration"]
        validation = validate_records(
            records,
            exact.input_len,
            args.gsp_output_len,
            config["prefix_tokens"],
            config["dp_size"],
            exact.tolerance_tokens,
            exact.strict_cache_hit,
        )
        result.update(
            latencies=[r.get("latency", 0) for r in records],
            dp_ranks=[
                dp_rank(r.get("meta_info") or {}, config["dp_size"]) for r in records
            ],
            num_retractions=[
                (r.get("meta_info") or {}).get("num_retractions") for r in records
            ],
            exact_cache_configuration=config,
            cache_validation=validation,
        )
        replace_last_jsonl(args.output_file, result)
        status = (
            "FAIL"
            if not validation["valid"]
            else "WARN" if validation["deviating_requests"] else "PASS"
        )
        print(
            f"Cache validation: {status}, mode={validation['mode']}, checked={len(records)}, failed={validation['failed_requests']}, deviating={validation['deviating_requests']}",
            flush=True,
        )
        if validation["deviating_requests"] and not exact.strict_cache_hit:
            print(
                "WARNING: actual cache hits differ from the target; keeping performance results in report mode. Inspect actual hit rates and per-DP statistics.",
                flush=True,
            )
        if not validation["valid"]:
            raise CacheValidationError(
                f"Invalid cache workload; inspect cache_validation in {args.output_file}"
            )
        return result

    serving.orjson = MetadataJSON()
    serving.get_dataset = exact_dataset
    serving.ASYNC_REQUEST_FUNCS["sglang"] = measured_request
    serving.run_benchmark = run


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exact-input-len", dest="input_len", type=int, required=True)
    parser.add_argument(
        "--exact-hit-percent", dest="hit_percent", type=int, required=True
    )
    parser.add_argument(
        "--exact-num-requests", dest="num_requests", type=int, required=True
    )
    parser.add_argument("--exact-page-size", dest="page_size", default="auto")
    parser.add_argument("--exact-dp-size", dest="dp_size", default="auto")
    parser.add_argument("--strict-cache-hit", action="store_true")
    parser.add_argument(
        "--cache-hit-tolerance-tokens", dest="tolerance_tokens", type=int, default=0
    )
    parser.add_argument(
        "--exact-warmup-timeout", dest="warmup_timeout", type=float, default=3600
    )
    exact, remaining = parser.parse_known_args()
    if (
        exact.input_len <= 0
        or exact.num_requests <= 0
        or not 0 <= exact.hit_percent < 100
        or exact.tolerance_tokens < 0
        or exact.warmup_timeout <= 0
    ):
        parser.error("Invalid exact cache workload settings")
    sys.argv = [sys.argv[0]] + remaining
    # Import lazily so pure workload/validation helpers can be tested offline.
    try:
        from sglang.benchmark import serving

        install_adapter(serving, exact)
        serving.cli_main()
    except CacheValidationError as error:
        print(str(error), file=sys.stderr)
        return 4
    except Exception as error:
        print(f"Exact cache benchmark failed: {error}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
