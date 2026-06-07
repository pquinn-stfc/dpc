#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow

label: "DPC batch — process many NeXus scans in one run"
doc: |
  Scatters the dpc_tool over an array of NeXus files, so a single
  invocation processes a whole set of scans.  Each input File may carry a
  Directory secondaryFile (the Diamond external-data layout); these travel
  with each File through the scatter.

requirements:
  ScatterFeatureRequirement: {}

inputs:

  nexus_files:
    type: File[]
    label: "Input NeXus/HDF5 files"
    doc: "One or more master .nxs files, each with its data dir as secondaryFile."
    secondaryFiles:
      - pattern: "$(self.nameroot)"
        required: false

  mapping_yaml:
    type: File
    label: "YAML field-mapping file (shared by all scans)"

  method:
    type:
      type: enum
      symbols: [kottler, arnison, frankot, ishizuka, scs]
    default: kottler

  crop_size:
    type: int
    default: 256

  format:
    type: string[]
    default: ["hdf5"]

  full_mask:
    type: boolean
    default: false

  no_phase:
    type: boolean
    default: false

  debug:
    type: boolean
    default: false

# ---------------------------------------------------------------------------
# Scatter the single-file tool over every input file
# ---------------------------------------------------------------------------
steps:

  dpc:
    run: dpc_tool.cwl
    scatter: nexus_file
    in:
      nexus_file: nexus_files
      mapping_yaml: mapping_yaml
      method: method
      crop_size: crop_size
      format: format
      full_mask: full_mask
      no_phase: no_phase
      debug: debug
    out: [result_nxs, result_images, debug_figure]

# ---------------------------------------------------------------------------
# Collect the per-scan outputs into arrays
# ---------------------------------------------------------------------------
outputs:

  results_nxs:
    type: File[]?
    outputSource: dpc/result_nxs

  results_images:
    type:
      type: array
      items: [ "null", { type: array, items: File } ]
    outputSource: dpc/result_images

  debug_figures:
    type: File[]?
    outputSource: dpc/debug_figure
