"""Project settings.

Set the two paths below to the locations of your Lhotse manifests and derived
paper artefacts.  They deliberately have no institution-specific defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


FRAME_SHIFT = 0.02  # 20 ms
RANK = 16
DEVICE = os.environ.get("PHONE_ASSESSMENT_DEVICE", "cuda")

# Configure these before running the notebooks.
MANIFEST_PATH = Path(os.environ.get("PHONE_ASSESSMENT_MANIFEST_PATH", "data/manifests"))
COMMON_PATH = Path(os.environ.get("PHONE_ASSESSMENT_DATA_PATH", "data"))

SANDI_CEFR_DICT = {
    "a2": 2.0,
    "a2+": 2.5,
    "b1": 3.0,
    "b1+": 3.5,
    "b2": 4.0,
    "b2+": 4.5,
    "c1": 5.0,
    "c2": 5.5,
}

UMEERJ_LEVELS_DICT = {str(level): level for level in range(6)}
UMEERJ_INVALID_SCORES = (-1, 0)

MODELS = {
    "wavlm_base_plus": {
        "hf_name": "microsoft/wavlm-base-plus",
        "layer": [6, 9],
    },
}
