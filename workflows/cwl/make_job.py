#!/usr/bin/env python3
'''Generate a CWL job YAML for the DPC tool from scan file numbers.

All Diamond I14 scans follow the naming convention ``i14-<NNNNN>.nxs`` with
the external detector data in a sibling directory ``i14-<NNNNN>/``.  Rather
than writing the explicit File/Directory paths by hand, this script builds
the job YAML from one or more file numbers and a data root.

Single scan  → a job for cwl/dpc_tool.cwl
Multiple     → a job for cwl/dpc_batch.cwl (scatter)

Usage
-----
    # one scan
    python cwl/make_job.py --data-root ~/Downloads 264401 > job.yml
    cwltool cwl/dpc_tool.cwl job.yml

    # a batch
    python cwl/make_job.py --data-root ~/Downloads 264401 264402 264403 > batch.yml
    cwltool cwl/dpc_batch.cwl batch.yml

    # a contiguous range (inclusive)
    python cwl/make_job.py --data-root ~/Downloads --range 264401 264405 > batch.yml

Options mirror the tool inputs: --method, --crop-size, --format, --debug,
--full-mask, --no-phase, --mapping.
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PREFIX = "i14-"


def build_file_entry(data_root: Path, fileno: str) -> dict:
    '''Build a CWL File object for one scan number, attaching the external
    data directory as a secondaryFile when it exists.'''
    stem = f"{PREFIX}{fileno}"
    nxs = data_root / f"{stem}.nxs"
    if not nxs.exists():
        sys.exit(f"error: {nxs} does not exist")

    entry = {"class": "File", "path": str(nxs.resolve())}

    data_dir = data_root / stem
    if data_dir.is_dir():
        entry["secondaryFiles"] = [
            {"class": "Directory", "path": str(data_dir.resolve())}
        ]
    return entry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("filenos", nargs="*",
                   help="One or more scan numbers, e.g. 264401 264402")
    p.add_argument("--range", nargs=2, metavar=("FIRST", "LAST"), type=int,
                   help="Inclusive range of scan numbers")
    p.add_argument("--data-root", required=True, type=Path,
                   help="Directory containing the i14-*.nxs files")
    p.add_argument("--mapping", type=Path,
                   default=Path(__file__).resolve().parents[2] /
                           "config" / "i14_264401_mapping.yaml",
                   help="Field-mapping YAML (default: config/i14_264401_mapping.yaml)")
    p.add_argument("--method", default="kottler",
                   choices=["kottler", "arnison", "frankot", "ishizuka", "scs"])
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--format", nargs="+", default=["hdf5"],
                   choices=["hdf5", "png", "tif"])
    p.add_argument("--full-mask", action="store_true")
    p.add_argument("--no-phase", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--batch", action="store_true",
                   help="Force the batch (nexus_files array) form even for a "
                        "single scan — use with cwl/dpc_batch.cwl")
    args = p.parse_args(argv)

    # Assemble the list of scan numbers
    filenos = list(args.filenos)
    if args.range:
        first, last = args.range
        filenos += [str(n) for n in range(first, last + 1)]
    if not filenos:
        p.error("provide at least one file number, or use --range")

    # Shared parameters
    common = {
        "mapping_yaml": {"class": "File", "path": str(args.mapping.resolve())},
        "method": args.method,
        "crop_size": args.crop_size,
        "format": args.format,
        "full_mask": args.full_mask,
        "no_phase": args.no_phase,
        "debug": args.debug,
    }

    if len(filenos) == 1 and not args.batch:
        # Single-file job for dpc_tool.cwl
        job = {"nexus_file": build_file_entry(args.data_root, filenos[0]),
               **common}
    else:
        # Batch job for dpc_batch.cwl
        job = {"nexus_files": [build_file_entry(args.data_root, n)
                               for n in filenos],
               **common}

    yaml.safe_dump(job, sys.stdout, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
