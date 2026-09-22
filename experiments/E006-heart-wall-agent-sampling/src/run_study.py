from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_executable(provider: str) -> str:
    found = shutil.which(provider)
    if found:
        return found
    if provider == "claude":
        candidates = [
            Path.home() / ".local" / "bin" / "claude.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Microsoft"
            / "WinGet"
            / "Links"
            / "claude.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(f"Could not find {provider!r} CLI")


def clean_child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_EFFORT_LEVEL",
    ):
        env.pop(key, None)
    env["NO_COLOR"] = "1"
    return env


def parse_codex(stdout: str) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    final_text: str | None = None
    usage: dict[str, Any] | None = None
    parse_errors: list[str] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"line {line_number}: {exc}")
            continue
        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                final_text = item.get("text")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage")
    return final_text, usage, parse_errors


def parse_claude(stdout: str) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, None, None, [str(exc)]
    return payload.get("result"), payload.get("usage"), payload.get("modelUsage"), []


def build_command(
    provider: str,
    executable: str,
    model: str,
    effort: str,
    prompt: str,
    sandbox: Path,
) -> list[str]:
    if provider == "codex":
        return [
            executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "-C",
            str(sandbox),
            "-m",
            model,
            "-c",
            f"model_reasoning_effort={effort}",
            "-c",
            "approval_policy=never",
            "-c",
            "project_doc_max_bytes=0",
            "--disable",
            "memories",
            "--disable",
            "hooks",
            "--disable",
            "plugins",
            "--disable",
            "apps",
            "--enable",
            "skip_host_skill_discovery",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
            prompt,
        ]
    if provider == "claude":
        return [
            executable,
            "-p",
            prompt,
            "--model",
            model,
            "--effort",
            effort,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--safe-mode",
            "--tools",
            "",
            "--max-turns",
            "1",
        ]
    raise ValueError(f"Unsupported provider: {provider}")


def run_job(
    job: dict[str, Any],
    executables: dict[str, str],
    prompt: str,
    sandbox: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    provider = job["provider"]
    command = build_command(
        provider,
        executables[provider],
        job["model"],
        job["effort"],
        prompt,
        sandbox,
    )
    started_at = utc_now()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=sandbox,
            env=clean_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        exit_code = None
        timed_out = True

    if provider == "codex":
        final_text, usage, parse_errors = parse_codex(stdout)
        model_usage = None
    else:
        final_text, usage, model_usage, parse_errors = parse_claude(stdout)

    return {
        **job,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": round(time.perf_counter() - start, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "final_text": final_text,
        "usage": usage,
        "model_usage": model_usage,
        "parse_errors": parse_errors,
        "stdout": stdout,
        "stderr": stderr,
    }


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(json.loads(line)["job_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", choices=("codex", "claude"), action="append")
    parser.add_argument("--limit", type=int, help="Run at most this many missing jobs (pilot only).")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = set(args.provider or config["providers"].keys())
    executables = {provider: find_executable(provider) for provider in selected}

    sandbox = Path(tempfile.gettempdir()) / "heart-wall-agent-study-empty"
    sandbox.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for provider, provider_config in config["providers"].items():
        if provider not in selected:
            continue
        for effort in provider_config["efforts"]:
            for index in range(config["sample_size_per_condition"]):
                jobs.append(
                    {
                        "job_id": f"{provider}-{provider_config['model']}-{effort}-{index + 1:03d}",
                        "provider": provider,
                        "model": provider_config["model"],
                        "effort": effort,
                        "sample_index": index + 1,
                        "config_sha256": sha256_file(args.config),
                    }
                )

    random.Random(config["shuffle_seed"]).shuffle(jobs)
    completed_ids = load_completed_ids(args.output)
    jobs = [job for job in jobs if job["job_id"] not in completed_ids]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    print(f"Running {len(jobs)} missing jobs with {config['max_concurrency']} workers", flush=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
            futures = {
                pool.submit(
                    run_job,
                    job,
                    executables,
                    config["prompt"],
                    sandbox,
                    config["timeout_seconds"],
                ): job
                for job in jobs
            }
            for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                with write_lock:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                status = "ok" if result["exit_code"] == 0 and result["final_text"] else "failed"
                print(
                    f"[{completed_count}/{len(jobs)}] {result['job_id']} {status} {result['duration_seconds']:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
