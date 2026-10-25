#!/usr/bin/env python3
"""
Concurrent Responses API runner (asyncio + Fire) with error tolerance + progress.

- Reads system prompt from a .txt file
- Reads user prompts (one per line) from a .txt file
- Calls OpenAI Responses API concurrently (max concurrency configurable)
- Retries retryable errors with exponential backoff + jitter (429, 5xx, timeouts)
- Logs each result as one JSON object per line (JSONL)
- Shows progress + prints each user prompt as the call starts

Requires:
  pip install openai fire
Env:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire
from openai import AsyncOpenAI

# Optional typed exceptions (names vary slightly across SDK versions).
try:
    from openai import (
        RateLimitError,
        APIError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )
except Exception:  # pragma: no cover
    RateLimitError = APIError = APITimeoutError = APIConnectionError = InternalServerError = Exception  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip("\n")


def _read_lines(path: Path) -> List[str]:
    # Keep line-by-line prompts (ignore totally empty lines)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.strip() != ""]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    model: str
    system_path: Path
    user_path: Path
    out_jsonl: Path
    concurrency: int
    reasoning_effort: str
    max_output_tokens: int

    # Error tolerance / retry controls
    max_retries: int
    timeout_s: float
    base_backoff_s: float
    max_backoff_s: float

    # Progress / logging
    show_prompts: bool
    prompt_preview_chars: int


def _get_status_code(exc: Exception) -> Optional[int]:
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc2 = getattr(resp, "status_code", None)
        if isinstance(sc2, int):
            return sc2
    return None


def _is_retryable(exc: Exception) -> Tuple[bool, str]:
    if isinstance(exc, RateLimitError):
        return True, "rate_limit"

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True, "network_or_timeout"

    if isinstance(exc, (InternalServerError, APIError)):
        sc = _get_status_code(exc)
        if sc is None:
            return True, "api_error_unknown_status"
        if sc >= 500:
            return True, f"server_{sc}"
        if sc == 429:
            return True, "rate_limit_429"

    sc = _get_status_code(exc)
    if sc == 429:
        return True, "rate_limit_429"
    if sc is not None and sc >= 500:
        return True, f"server_{sc}"

    if isinstance(exc, TimeoutError):
        return True, "timeout"

    return False, "non_retryable"


def _backoff_delay(attempt: int, base_s: float, max_s: float) -> float:
    exp = base_s * (2 ** (attempt - 1))
    return random.uniform(0, min(max_s, exp))


def _preview(s: str, n: int) -> str:
    if n <= 0:
        return s
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


class App:
    def run(
        self,
        model: str = "gpt-5.2",
        system_path: str = "data/system_prompt.txt",
        user_path: str = "data/user_prompts.txt",
        out_jsonl: str = "runs/output.jsonl",
        concurrency: int = 20,
        reasoning_effort: str = "none",
        max_output_tokens: int = 8192,
        # error tolerance knobs
        max_retries: int = 6,
        timeout_s: float = 90.0,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 20.0,
        # progress knobs
        show_prompts: bool = True,
        prompt_preview_chars: int = 200,
    ) -> Dict[str, Any]:
        cfg = RunConfig(
            model=model,
            system_path=Path(system_path),
            user_path=Path(user_path),
            out_jsonl=Path(out_jsonl),
            concurrency=int(concurrency),
            reasoning_effort=str(reasoning_effort),
            max_output_tokens=int(max_output_tokens),
            max_retries=int(max_retries),
            timeout_s=float(timeout_s),
            base_backoff_s=float(base_backoff_s),
            max_backoff_s=float(max_backoff_s),
            show_prompts=bool(show_prompts),
            prompt_preview_chars=int(prompt_preview_chars),
        )

        if cfg.concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        if not cfg.system_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {cfg.system_path}")

        if not cfg.user_path.exists():
            raise FileNotFoundError(f"User prompts file not found: {cfg.user_path}")

        _ensure_parent_dir(cfg.out_jsonl)
        return asyncio.run(_run_async(cfg))


async def _run_async(cfg: RunConfig) -> Dict[str, Any]:
    system_prompt = _read_text(cfg.system_path)
    user_prompts = _read_lines(cfg.user_path)
    if not user_prompts:
        raise ValueError(f"No prompts found in {cfg.user_path}")

    total = len(user_prompts)
    sem = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    # Counters for progress
    counters_lock = asyncio.Lock()
    started = 0
    done = 0
    ok = 0
    failed = 0

    client = AsyncOpenAI()

    started_at = _utc_now_iso()
    t_all0 = time.perf_counter()

    f = cfg.out_jsonl.open("a", encoding="utf-8")

    async def _print(msg: str) -> None:
        async with print_lock:
            print(msg, flush=True)

    async def _progress_line(prefix: str = "") -> None:
        async with counters_lock:
            s, d, o, e = started, done, ok, failed
        elapsed = time.perf_counter() - t_all0
        rate = d / elapsed if elapsed > 0 else 0.0
        await _print(
            f"{prefix}progress: started={s}/{total} done={d}/{total} ok={o} failed={e} "
            f"rate={rate:.2f}/s elapsed={elapsed:.1f}s"
        )

    async def create_with_retries(user_prompt: str) -> Any:
        for attempt in range(0, cfg.max_retries + 1):
            try:
                coro = client.responses.create(
                    model=cfg.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    reasoning={"effort": cfg.reasoning_effort},
                    max_output_tokens=cfg.max_output_tokens,
                )
                return await asyncio.wait_for(coro, timeout=cfg.timeout_s)
            except Exception as e:
                retryable, _reason = _is_retryable(e)
                if attempt >= cfg.max_retries or not retryable:
                    raise
                delay = _backoff_delay(
                    attempt=attempt + 1,
                    base_s=cfg.base_backoff_s,
                    max_s=cfg.max_backoff_s,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Unreachable: retries exhausted")  # safety

    async def call_one(i: int, user_prompt: str) -> None:
        nonlocal started, done, ok, failed

        async with sem:
            async with counters_lock:
                started += 1
                s_now = started

            if cfg.show_prompts:
                await _print(
                    f"[START {i}] ({s_now}/{total}) user_prompt={_preview(user_prompt, cfg.prompt_preview_chars)}"
                )

            t0 = time.perf_counter()

            record: Dict[str, Any] = {
                "index": i,
                "started_at": started_at,
                "logged_at": _utc_now_iso(),
                "request": {
                    "model": cfg.model,
                    "instructions": system_prompt,
                    "input": user_prompt,
                    "reasoning": {"effort": cfg.reasoning_effort},
                    "max_output_tokens": cfg.max_output_tokens,
                    "timeout_s": cfg.timeout_s,
                    "max_retries": cfg.max_retries,
                    "backoff": {"base_s": cfg.base_backoff_s, "max_s": cfg.max_backoff_s},
                },
            }

            try:
                resp = await create_with_retries(user_prompt)
                assistant_text = getattr(resp, "output_text", None)

                record["conversation"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
                record["response"] = resp.model_dump()
                record["request_id"] = getattr(resp, "_request_id", None)
                record["status"] = getattr(resp, "status", None)
                record["duration_s"] = round(time.perf_counter() - t0, 6)

                async with counters_lock:
                    ok += 1

                await _print(f"[DONE  {i}] ok duration_s={record['duration_s']}")

            except Exception as e:
                record["error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "status_code": _get_status_code(e),
                    "retryable": _is_retryable(e)[0],
                    "traceback": traceback.format_exc(),
                }
                record["duration_s"] = round(time.perf_counter() - t0, 6)

                async with counters_lock:
                    failed += 1

                await _print(
                    f"[DONE  {i}] FAILED duration_s={record['duration_s']} "
                    f"type={record['error']['type']} status={record['error']['status_code']}"
                )

            async with counters_lock:
                done += 1
                d_now = done

            # Occasionally print a progress summary
            if d_now % 25 == 0 or d_now == total:
                await _progress_line(prefix="   ")

            # Write JSONL
            line = json.dumps(record, ensure_ascii=False)
            async with write_lock:
                f.write(line + "\n")
                f.flush()

    try:
        await _print(f"Starting run: total={total} concurrency={cfg.concurrency} model={cfg.model}")
        tasks = [asyncio.create_task(call_one(i, p)) for i, p in enumerate(user_prompts)]
        await asyncio.gather(*tasks)
    finally:
        f.close()
        try:
            await client.close()
        except Exception:
            pass

    ended_at = _utc_now_iso()
    elapsed = time.perf_counter() - t_all0

    # Final progress
    await _progress_line(prefix="FINAL ")

    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "count": total,
        "out_jsonl": str(cfg.out_jsonl),
        "model": cfg.model,
        "concurrency": cfg.concurrency,
        "reasoning_effort": cfg.reasoning_effort,
        "max_output_tokens": cfg.max_output_tokens,
        "max_retries": cfg.max_retries,
        "timeout_s": cfg.timeout_s,
        "elapsed_s": round(elapsed, 6),
    }


if __name__ == "__main__":
    fire.Fire(App)
