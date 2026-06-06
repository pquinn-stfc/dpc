# Running DPC via CWL

The DPC tool is wrapped as a [Common Workflow Language](https://www.commonwl.org/)
`CommandLineTool` so it can run unchanged under any CWL-compliant runner.
Two runners are tested here: **cwltool** (the reference implementation) and
**Toil** (a scalable runner that also targets HPC and cloud batch systems).

## Files

| File | Purpose |
|------|---------|
| `dpc_tool.cwl` | The CWL tool definition |
| `i14_264401_job.yml` | Example job input (the i14-264401 scan) |
| `run_toil.sh` | Convenience wrapper for `toil-cwl-runner` |

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
cwltool --outdir ./cwl_output cwl/dpc_tool.cwl cwl/i14_264401_job.yml
```

## Run with Toil

```bash
# Convenience wrapper (handles PATH + flags):
cwl/run_toil.sh ./toil_output

# Or directly:
toil-cwl-runner \
    --jobStore ./toil_jobstore \
    --outdir ./toil_output \
    --no-prepull \
    --clean always \
    cwl/dpc_tool.cwl cwl/i14_264401_job.yml
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
