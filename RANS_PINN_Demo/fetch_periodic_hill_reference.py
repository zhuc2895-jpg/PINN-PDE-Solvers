"""Download the independent ERCOFTAC case-081 periodic-hill LES archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from .periodic_hill import download_reference

__all__ = ["main"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("RANS_PINN_Demo/reference/periodic_hill"),
        help="directory that receives the archive and extracted profiles",
    )
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extracted = download_reference(args.destination, force=args.force)
    print(f"Extracted ERCOFTAC case-081 reference to {extracted}")


if __name__ == "__main__":  # pragma: no cover
    main()
