# Running DPC via CWL

The DPC tool is wrapped as a [Common Workflow Language](https://www.commonwl.org/)
`CommandLineTool` so it can run unchanged under any CWL-compliant runner.
Two runners are tested here: **cwltool** (the reference implementation) and
**Toil** (a scalable runner that also targets HPC and cloud batch systems).

## Files

| File | Purpose |
|------|---------|
| `dpc_tool.cwl` | Single-scan CWL tool definition |
| `dpc_batch.cwl` | Workflow that scatters the tool over many scans |
| `make_job.py` | Generate a job YAML from scan **file numbers** |
| `i14_264401_job.yml` | Hand-written example job (single scan) |
| `run_toil.sh` | Convenience wrapper for `toil-cwl-runner` |

## Generic usage — just pass file numbers

All I14 scans are named `i14-<NNNNN>.nxs` with data in a sibling
`i14-<NNNNN>/` directory.  `make_job.py` builds the job YAML from one or
more scan numbers plus a data root, so you never write paths by hand.

```bash
# One scan → job for dpc_tool.cwl
python workflows/cwl/make_job.py --data-root ~/Downloads 264401 --format hdf5 png > job.yml
cwltool --outdir ./out workflows/cwl/dpc_tool.cwl job.yml

# Several scans → job for dpc_batch.cwl (one run processes them all)
python workflows/cwl/make_job.py --data-root ~/Downloads 264401 264402 264403 > batch.yml
cwltool --outdir ./out workflows/cwl/dpc_batch.cwl batch.yml

# A contiguous range (inclusive)
python workflows/cwl/make_job.py --data-root ~/Downloads --range 264401 264410 > batch.yml
cwltool --outdir ./out workflows/cwl/dpc_batch.cwl batch.yml

# Force the batch (array) form for a single scan
python workflows/cwl/make_job.py --data-root ~/Downloads 264401 --batch > batch.yml
```

The script auto-selects the singular `nexus_file` form (for `dpc_tool.cwl`)
when there is one scan, or the `nexus_files` array form (for
`dpc_batch.cwl`) when there are several.  It attaches the external-data
directory as a `secondaryFile` automatically when it exists.

Processing options pass straight through:
`--method`, `--crop-size`, `--format`, `--full-mask`, `--no-phase`,
`--debug`, and `--mapping` (defaults to `config/i14_264401_mapping.yaml`).

Why a generator rather than a fully generic YAML?  CWL deliberately keeps
input files explicit (for provenance and portability) and does not let a
workflow synthesise arbitrary filesystem paths from a scalar like a scan
number.  The generator is the idiomatic way to get that ergonomics while
keeping the job file a faithful, reproducible record of exactly what ran.

## Prerequisites

Build the Docker image first (the CWL references it by `dockerImageId`):

```bash
docker build -t pquinn-stfc/dpc:latest .
```

## External detector data (Diamond layout)

Diamond NeXus files are *master* files: the small `.nxs` (~140 KB) holds
metadata and **external links** to per-detector `.h5` files in a sibling
directory named after the scan, e.g.:

```
i14-264401.nxs                                  ← master file (input)
i14-264401/
    i14-264401-merlin_addetector.h5             ← 862 MB detector data
    i14-264401-xsp3_addetector.h5
    ...
```

So the data directory must be staged alongside the master file.  The job
input declares it as a `secondaryFile` of class `Directory`:

```yaml
nexus_file:
  class: File
  path: /path/to/i14-264401.nxs
  secondaryFiles:
    - class: Directory
      path: /path/to/i14-264401
```

For self-contained NeXus files (data stored inline) the `secondaryFiles`
entry is simply omitted — the tool's `secondaryFiles` pattern is marked
`required: false`.

## Run with cwltool

```bash
cwltool --outdir ./cwl_output workflows/cwl/dpc_tool.cwl workflows/cwl/i14_264401_job.yml
```

## Run with Toil

```bash
# Convenience wrapper (handles PATH + flags):
workflows/cwl/run_toil.sh ./toil_output

# Or directly:
toil-cwl-runner \
    --jobStore ./toil_jobstore \
    --outdir ./toil_output \
    --no-prepull \
    --clean always \
    workflows/cwl/dpc_tool.cwl workflows/cwl/i14_264401_job.yml
```

### Batch with Toil

The batch workflow runs under Toil too — and this is where Toil shines, as
each scattered scan can be dispatched to a separate node on a cluster:

```bash
python workflows/cwl/make_job.py --data-root ~/Downloads --range 264401 264410 > batch.yml
toil-cwl-runner --jobStore ./js --outdir ./out --no-prepull --clean always \
    workflows/cwl/dpc_batch.cwl batch.yml
```

### Toil notes

- `--no-prepull` skips the container pre-pull step (which needs
  `cwl-docker-extract` from `cwl-utils`); the local image is used directly.
- Toil spawns a `_toil_worker` helper as a subprocess, so the directory where
  `pip install 'toil[cwl]'` placed its console scripts must be on `PATH`.
  `run_toil.sh` adds the pip user-bin directory automatically.
- The same `--jobStore` can be a cloud/HPC location (AWS, Google, Slurm,
  Kubernetes) to scale the identical workflow out — see the Toil docs.

## Output

Both runners produce identical results in the chosen `--outdir`:

```
i14-264401_dpc.nxs          NeXus result (phase, dx, dy, grad_norm, axes)
i14-264401_dpc_phase.png    phase preview
i14-264401_dpc_dx.png       phase gradient x
i14-264401_dpc_dy.png       phase gradient y
i14-264401_dpc_grad_norm.png
i14-264401_dpc_debug.png    diagnostic figure (when debug: true)
```
