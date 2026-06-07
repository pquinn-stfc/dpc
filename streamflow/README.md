# Multi-site execution with StreamFlow

[StreamFlow](https://streamflow.di.unito.it/) runs the **existing** DPC CWL
(`cwl/dpc_batch.cwl`) unchanged and adds a deployment layer that maps workflow
steps to different execution sites — here Google Cloud (GKE), an HPC Slurm
cluster, and an OpenStack cloud — within a single workflow.

> **Status:** `streamflow.yml` is a **template**.  It has not been run against
> live GCP/Slurm/OpenStack here.  Validate the connector field names against
> your installed StreamFlow version (`streamflow version`) — the schema has
> changed across releases.

## How it differs from Toil/cwltool

`cwltool` and Toil each target **one** execution backend per run.  StreamFlow
can bind *different steps* (or different instances of a scattered step) to
*different* sites in one workflow, and it **moves the data** between sites
automatically.  That makes genuine single-submission hybrid execution
possible while keeping your CWL portable.

## Prerequisites

### 1. Registry-hosted image (required for multi-site)

Every site must be able to pull the container.  The local CWL uses a
locally-built image:

```yaml
# cwl/dpc_tool.cwl (local — works only where the image was built)
DockerRequirement:
  dockerImageId: "pquinn-stfc/dpc:latest"
```

For multi-site, change this to a pinned registry tag so every site pulls an
identical image:

```yaml
DockerRequirement:
  dockerPull: "ghcr.io/pquinn-stfc/dpc:0.1.0"
```

Build and push once:

```bash
docker build -t ghcr.io/pquinn-stfc/dpc:0.1.0 .
docker push ghcr.io/pquinn-stfc/dpc:0.1.0
```

On HPC/OpenStack the image runs under **Singularity**; StreamFlow converts the
Docker image automatically (the host needs Singularity/Apptainer installed).

### 2. Site access configured locally

| Site | What StreamFlow needs |
|------|----------------------|
| GKE | kubeconfig — `gcloud container clusters get-credentials … --kubeconfig ~/.kube/gke-dpc.config` |
| Slurm | SSH access to the login node (`hostname`, `username`, `sshKey`) |
| OpenStack | a running VM with Singularity + your SSH key |

### 3. A job input file

```bash
python cwl/make_job.py --data-root ~/Downloads --range 264401 264410 --batch \
    > cwl/batch_job.yml
```

## Run

```bash
cd streamflow
streamflow run streamflow.yml
```

## The key practical caveat: data locality

Each scan's input is a small `.nxs` **plus ~862 MB** of external detector data.
StreamFlow stages a step's inputs to wherever that step runs, so freely
spreading the scatter across three clouds moves a lot of data over the wire.

Two models:

1. **Free spread** (what the template shows) — `/dpc` bound to all three
   sites; StreamFlow's scheduler balances instances across them and transfers
   data as needed.  Simple, but pays the transfer cost for off-site scans.

2. **Data-local** (recommended for production) — one StreamFlow file per site,
   each binding `/dpc` to that site only, over the subset of scans whose data
   already lives there.  Keeps the big detector files next to the compute.
   The commented block at the end of `streamflow.yml` shows the per-site
   binding form.

For Diamond data (which originates at Diamond), the data-local model — or
simply running the Toil partition pattern per site — usually wins unless you
specifically need to burst compute to a cloud.

## When to actually reach for this

StreamFlow earns its keep when a *single workflow* has steps that must run in
*different* places (e.g. reconstruct on HPC, then post-process on a GPU cloud).
For the DPC batch — an embarrassingly parallel scatter with no cross-step data
dependencies — the simpler **Toil partition-per-site** approach
(`cwl/run_toil.sh` over a per-site scan range) is less machinery for the same
result.  Use StreamFlow here mainly if you want one submission and StreamFlow's
automatic cross-site data handling.
