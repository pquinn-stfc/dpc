#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool

label: "DPC — Differential Phase Contrast processing tool"
doc: |
  Computes differential phase contrast images from a NeXus/HDF5 pixelated
  detector dataset.  Outputs a NeXus result file and optional PNG previews.

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
requirements:
  DockerRequirement:
    dockerImageId: "pquinn-stfc/dpc:latest"
  ResourceRequirement:
    ramMin: 2048    # MB — lazy loading keeps peak RAM low, but leave headroom
    coresMin: 1
  InlineJavascriptRequirement: {}

baseCommand: ["python", "/app/main.py"]

# All outputs are written next to the input file by default, but we force
# everything into the output directory by fixing --output to $(runtime.outdir).
arguments:
  - prefix: "--output"
    valueFrom: "$(runtime.outdir)/$(inputs.nexus_file.nameroot)_dpc"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
inputs:

  nexus_file:
    type: File
    label: "Input NeXus/HDF5 file"
    doc: |
      The NeXus master file.  At Diamond, detector data is held in external
      .h5 files inside a sibling directory named after the scan (the file's
      nameroot).  Declare that directory as a secondaryFile of class Directory
      in the job input so the runner stages it alongside the master file.
      For self-contained NeXus files no secondaryFile is needed.
    secondaryFiles:
      - pattern: "$(self.nameroot)"
        required: false
    inputBinding:
      position: 1

  mapping_yaml:
    type: File
    label: "YAML field-mapping file"
    doc: |
      Declares how HDF5 dataset paths map to DPCDataset fields.
      See config/i14_264401_mapping.yaml for an example.
    inputBinding:
      position: 2

  method:
    type:
      type: enum
      symbols: [kottler, arnison, frankot, ishizuka, scs]
    default: kottler
    label: "Phase retrieval method"
    inputBinding:
      prefix: "--method"

  crop_size:
    type: int?
    default: 256
    label: "Detector crop size in pixels"
    inputBinding:
      prefix: "--crop-size"

  format:
    type: string[]
    default: ["hdf5"]
    label: "Output formats"
    doc: "One or more of: hdf5, png, tif"
    inputBinding:
      prefix: "--format"

  full_mask:
    type: boolean?
    default: false
    label: "Use full pixel-masking mode"
    doc: "Slower but more reliable — computes mean/std across all frames."
    inputBinding:
      prefix: "--full-mask"

  no_phase:
    type: boolean?
    default: false
    label: "Skip phase retrieval"
    doc: "Only compute gradient norm; useful for quick QC."
    inputBinding:
      prefix: "--no-phase"

  debug:
    type: boolean?
    default: false
    label: "Save diagnostic figure"
    doc: "Writes <stem>_debug.png showing masks and overlays."
    inputBinding:
      prefix: "--debug"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
outputs:

  result_nxs:
    type: File?
    label: "NeXus result file"
    doc: "NXdpc-schema HDF5 containing phase, dx, dy, grad_norm and axes."
    outputBinding:
      glob: "$(inputs.nexus_file.nameroot)_dpc.nxs"

  result_images:
    type: File[]?
    label: "PNG/TIFF preview images"
    outputBinding:
      glob:
        - "$(inputs.nexus_file.nameroot)_dpc_*.png"
        - "$(inputs.nexus_file.nameroot)_dpc_*.tif"

  debug_figure:
    type: File?
    label: "Diagnostic debug figure"
    outputBinding:
      glob: "$(inputs.nexus_file.nameroot)_dpc_debug.png"
