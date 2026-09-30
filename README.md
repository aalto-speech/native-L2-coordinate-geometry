# Phone-based assessment: paper code

This folder is the compact code release for the paper.  It contains the shared
implementation used by the analysis notebooks, but excludes intermediate
pickle files, trained-model outputs, notebook checkpoints, and earlier
experiments.

## Layout

- `src/build_phone_codes.py` builds phone-unit codebooks and averages.
- `src/distance_calculation.py` calculates SANDI and UME-ERJ distances.
- `src/distance_evaluation_sandi.py` evaluates SANDI part-level CEFR
  associations.
- `src/distance_evaluation_umeerj.py` evaluates UME-ERJ recording-level,
  rater-aware associations.
- `src/utils.py` and `src/projections.py` provide shared helpers.

The workflow examples remain in the repository's `notebooks/` directory:
`codebook.ipynb`, `avg.ipynb`, `calculate_distance.ipynb`,
`calculate_distance_random.ipynb`, `eval_distance.ipynb`, and
`eval_distance_random.ipynb`.

## Setup

Install the dependencies and add `src` to Python's import path:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Configure data locations with environment variables before opening a notebook:

```bash
export PHONE_ASSESSMENT_MANIFEST_PATH=/path/to/lhotse_manifests
export PHONE_ASSESSMENT_DATA_PATH=/path/to/paper_data
```

`src/config.py` contains the remaining analysis settings.  The release does
not include audio, Lhotse manifests, alignments, feature matrices, codebooks,
or distance pickles; obtain and place those data according to their respective
data-use agreements.

## Evaluation imports

The former mixed `distance_evaluation.py` module has been split.  Import the
dataset-specific evaluator explicitly:

```python
from distance_evaluation_sandi import analyze_all_distance_associations
from distance_evaluation_umeerj import umeerj_evaluate_distance_grid_clean
```

This keeps the SANDI CEFR workflow separate from the UME-ERJ rater-aware
workflow and avoids distributing duplicate evaluator code.
