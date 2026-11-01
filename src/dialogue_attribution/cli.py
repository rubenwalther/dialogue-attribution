"""Command line: segments | baseline | run | eval."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .attribute import attribute
from .baseline import baseline
from .evaluate import load_gold, load_lines, report, save_lines, score
from .llm import CachedLLM, CacheMiss, ChatClient, LLMError, model_slug
from .schema import CharacterList
from .segment import segment


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _characters(path: str) -> CharacterList:
    return CharacterList.model_validate_json(_read(path))


def _resolve_model(arg: str | None, cache: Path) -> str:
    """--model, else LLM_MODEL, else the only model folder in the cache."""
    if arg or os.environ.get("LLM_MODEL"):
        return arg or os.environ["LLM_MODEL"]
    folders = [p for p in cache.glob("*") if p.is_dir()] if cache.exists() else []
    files = sorted(folders[0].glob("*.json")) if len(folders) == 1 else []
    if files:
        return json.loads(files[0].read_text(encoding="utf-8"))["model"]
    raise SystemExit(f"No model found in {cache}/. Pass --model or set LLM_MODEL.")


def _git_commit() -> str | None:
    """Short commit hash of this repository, with "+dirty" if files were changed."""
    here = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                                 capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return (commit + ("+dirty" if changed else "")) or None


def cmd_segments(args: argparse.Namespace) -> None:
    for s in segment(_read(args.text)):
        if s.kind == "quote" or args.all:
            print(f"{s.index:4d} {s.kind:9s} {s.text}")


def cmd_baseline(args: argparse.Namespace) -> None:
    lines = baseline(segment(_read(args.text)), _characters(args.characters).names())
    save_lines(lines, Path(args.out))
    print(f"Wrote {len(lines)} lines to {args.out}")


def cmd_run(args: argparse.Namespace) -> None:
    cache = Path(args.cache)
    model = _resolve_model(args.model, cache)
    try:
        client = ChatClient.from_env(model) if args.live else None
        llm = CachedLLM(cache_dir=cache / model_slug(model), model=model, client=client)
        segments = segment(_read(args.text))
        lines = attribute(segments, _characters(args.characters), llm, max_chars=args.max_chars)
    except (CacheMiss, LLMError, ValueError) as err:
        raise SystemExit(f"Error: {err}") from err
    save_lines(lines, Path(args.out))
    u = llm.usage
    meta = {
        "model": model,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "quotes": sum(1 for line in lines if line.kind == "quote"),
        "live_requests": u.requests,
        "cache_hits": u.cache_hits,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "seconds": round(u.seconds, 1),
    }
    # Only a complete live run measures time and tokens for the whole chapter.
    if u.requests and not u.cache_hits:
        Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                            encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {args.out} "
          f"({u.requests} live requests, {u.cache_hits} from cache)")


def cmd_eval(args: argparse.Namespace) -> None:
    try:
        result = score(load_lines(args.predicted), load_gold(args.gold))
    except ValueError as err:
        raise SystemExit(f"Error: {err}") from err
    text = report(args.name or Path(args.predicted).stem, result)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dialogue-attribution",
                                     description="Who says each quote, and how?")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("segments", help="print the quotes the segmenter finds")
    p.add_argument("text")
    p.add_argument("--all", action="store_true", help="also print narration")
    p.set_defaults(func=cmd_segments)

    p = sub.add_parser("baseline", help="label quotes with simple rules (no API key)")
    p.add_argument("text")
    p.add_argument("--characters", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("run", help="label quotes with a language model (cache first)")
    p.add_argument("text")
    p.add_argument("--characters", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--model", help="default: LLM_MODEL, or the only model in the cache")
    p.add_argument("--live", action="store_true", help="call the API for answers not in the cache")
    p.add_argument("--max-chars", type=int, default=20000, help="characters per request")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eval", help="compare labels with gold labels")
    p.add_argument("predicted")
    p.add_argument("gold")
    p.add_argument("--name")
    p.add_argument("--report", help="append the Markdown summary to this file")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)
