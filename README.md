# Native-reference coordinate geometry for L2 pronunciation analysis

This repository contains research code and example Jupyter notebooks for
building native-reference phone representations, calculating distances for
L2 speech, and evaluating their associations with proficiency or
pronunciation ratings.

The work is described in these papers:

- Initial study: [A Native-Reference Coordinate Geometry for L2 Pronunciation
  Deviation Using Self-Supervised Speech Models](https://arxiv.org/abs/2609.28060).
- Extended study: [A Native-Reference Phone-Class Geometry for Second-Language
  Pronunciation Analysis](https://arxiv.org/abs/2609.30075).

## Repository contents

- `src/build_phone_codes.py` builds phone-class averages and reference codebooks.
- `src/distance_calculation.py` calculates SANDI and UME-ERJ distances.
- `src/distance_evaluation_sandi.py` evaluates SANDI distance associations.
- `src/distance_evaluation_umeerj.py` evaluates UME-ERJ results.
- `src/utils.py`, `src/projections.py`, and `src/config.py` contain shared
  helpers, projection methods, and configuration.
- `example_notebooks/` contains workflow examples:
  - `feature_extraction.ipynb`
  - `phone_averaging.ipynb`
  - `native_reference_space.ipynb`
  - `calculate_distance.ipynb`
  - `eval_distance.ipynb`

## Example notebooks

The notebooks demonstrate the workflow and call the code in `src/`, but they
are examples rather than turnkey scripts. They will need to be adapted to the
data available to you. In particular, review the paths, corpus and split
names, Lhotse cut sets, alignments, recording selections, model and layer
choices, and output locations before running them. Some cells contain path
placeholders for users to replace. Run the stages that create the required
inputs before running later stages.

The examples cover these stages:

1. Extract frame-level features and save them in pickle shards.
2. Average features over aligned phone classes.
3. Build native-reference codebooks from LBS features and alignments.
4. Calculate L2-to-reference distances.
5. Evaluate distance associations.

The examples use different corpora and model settings to illustrate the
workflow. Make the model, layer, split, alignment, rank, and strategy settings
consistent across the stages you choose to run.

## Data and access

Audio, complete Lhotse manifests, feature matrices, and most intermediate
pickles are not included in this Git repository. The `data/` directory is
Git-ignored and is intended for local data and generated artifacts. Set
`PHONE_ASSESSMENT_DATA_PATH` to the data root used by your workflow.

The workflow uses alignment and LBS recording-ID selection artifacts
associated with the official Lhotse preparation. These are inputs to the
notebooks; obtain or prepare the versions appropriate for your data and place
them under the configured data root. Any small preparation files distributed
separately with a release should be placed there as well.

UME-ERJ data are not included. Access to the dataset must be requested from
the data consortium. Follow the consortium's access conditions and data-use
agreement for the data, alignments, and any derived artifacts.

Do not publish or redistribute restricted data or derived files unless the
relevant data-use terms permit it. For large derived pickles, use approved
research data storage rather than committing them to Git. Keep a record of
the data version, model and layer, alignment source, and processing settings
used to generate results.

## Setup

Install the Python dependencies from the repository root and make `src/`
available to notebook imports:

```bash
pip install -r requirements.txt
pip install jupyterlab transformers
export PYTHONPATH="$PWD/src"
```

`transformers` is used by the feature-extraction example. The model weights
listed in `src/config.py` must also be available to the environment running
that notebook.

Set data locations before starting Jupyter. `PHONE_ASSESSMENT_DATA_PATH`
should point to the root containing (or where you will create) directories
such as `alignments/`, `features/`, and `mix/`. `PHONE_ASSESSMENT_MANIFEST_PATH`
should point to the Lhotse manifest root. By default, these are `data/` and
`data/manifests/` relative to the current working directory.

```bash
export PHONE_ASSESSMENT_DATA_PATH="/path/to/project-data"
export PHONE_ASSESSMENT_MANIFEST_PATH="/path/to/lhotse-manifests"
# Optional: defaults to cuda
export PHONE_ASSESSMENT_DEVICE="cpu"
jupyter lab
```

The manifest helper expects files under
`<manifest-root>/<corpus>/`, named like
`<partition>_recording_set.jsonl.gz` and
`<partition>_supervision_set.jsonl.gz`. Adjust the notebook loading code if
your manifests use a different layout.

The values in `src/config.py` are read when Python imports the module. If you
change an environment variable after opening a notebook, restart its kernel
and import the configuration again.

## Data flow

The notebooks save intermediate outputs under `PHONE_ASSESSMENT_DATA_PATH`.
The exact model, layer, corpus, and split depend on the notebook configuration.
Common output directories include:

- `features/` for extracted feature shards.
- `phone_avg/` for aligned phone-average pickles.
- `codebooks/` for native-reference codebooks.
- `distances/` for calculated distance results.

These generated files can be large and are ignored by Git when stored under
`data/`. They should be regenerated from accessible inputs or stored in an
approved external archive when sharing results.
