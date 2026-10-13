"""Local-only command line interface. Never print raw histories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .collection import collect, discover, inventory, normalize_collection, validate_collection
from .curation import curate_collection, validate_curated_collection


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Collect a private, unlabeled OM source corpus.")
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("inventory", "collect"):
        sub = commands.add_parser(command)
        sub.add_argument("--pi-root", type=Path, default=Path.home() / ".pi/agent/sessions")
        sub.add_argument("--no-pi", action="store_true", help="Collect only explicitly selected Hatfield stores.")
        sub.add_argument("--hatfield-om-dir", action="append", type=Path, default=[],
                         help="Repeat for each OM directory. All its .sqlite stores are backed up.")
        sub.add_argument("--hatfield-session-root", action="append", type=Path, default=[],
                         help="Optional explicit roots, paired in order with OM directories.")
        if command == "collect":
            sub.add_argument("--output", type=Path, required=True, help="New private directory outside sources.")
            sub.add_argument("--resume", action="store_true",
                             help="Resume using its stored inventory, ignoring source-selection flags.")
    validation = commands.add_parser("validate")
    validation.add_argument("collection", type=Path)
    normalization = commands.add_parser("normalize", help="Reparse an existing frozen corpus, not live sources.")
    normalization.add_argument("collection", type=Path)
    normalization.add_argument("--output", type=Path, required=True)
    curation = commands.add_parser("curate", help="Select OM-bearing sessions with resolved content evidence.")
    curation.add_argument("collection", type=Path)
    curation.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            manifest = json.loads((args.collection.expanduser() / "manifest.json").read_text())
            report = (validate_curated_collection(args.collection)
                      if manifest.get("schema_version") == "om-curated-corpus/1"
                      else validate_collection(args.collection))
        elif args.command == "normalize":
            report = normalize_collection(args.collection, args.output)
        elif args.command == "curate":
            report = curate_collection(args.collection, args.output)
        else:
            if args.command == "collect" and args.resume:
                sources = json.loads((args.output.expanduser() / "collection-config.json").read_text())["sources"]
            else:
                sources = discover(None if args.no_pi else args.pi_root, args.hatfield_om_dir,
                                   args.hatfield_session_root)
            report = (inventory(sources) if args.command == "inventory" else
                      collect(sources, args.output, args.resume))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        # Native text and provider settings must not leak through parser exceptions.
        # Full context remains in local artifacts; fail without claiming completion.
        print(f"Collection failed ({type(error).__name__}). No model calls were made.", file=sys.stderr)
        return 1
