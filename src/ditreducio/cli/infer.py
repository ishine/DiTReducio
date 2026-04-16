from __future__ import annotations

import argparse

from ditreducio.cli.common import add_shared_arguments
from ditreducio.cli.common import prepare_adapter
from ditreducio.cli.common import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiTReducio inference CLI")
    add_shared_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    adapter = prepare_adapter(args)
    adapter.infer(delta=args.delta, track_flops=args.track_flops, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
