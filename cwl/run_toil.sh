#!/usr/bin/env bash
# Run the DPC CWL tool with Toil (toil-cwl-runner).
#
# Usage:
#   cwl/run_toil.sh [OUTDIR] [JOBSTORE]
#
# Defaults:
#   OUTDIR   = ./toil_output
#   JOBSTORE = ./toil_jobstore  (deleted before each run)
#
# Toil runs the same CWL as cwltool; this wrapper just sets the
# pip user-script directory on PATH (so _toil_worker is found) and
# passes the flags that suit a local single-machine Docker run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-$PWD/toil_output}"
JOBSTORE="${2:-$PWD/toil_jobstore}"

# Ensure the pip user bin dir (where toil installs _toil_worker) is on PATH
USER_BIN="$(python3 -c 'import site,os;print(os.path.join(site.getuserbase(),"bin"))')"
export PATH="$USER_BIN:$PATH"

rm -rf "$JOBSTORE"
mkdir -p "$OUTDIR"

exec toil-cwl-runner \
  --jobStore "$JOBSTORE" \
  --outdir "$OUTDIR" \
  --no-prepull \
  --clean always \
  "$HERE/dpc_tool.cwl" "$HERE/i14_264401_job.yml"
