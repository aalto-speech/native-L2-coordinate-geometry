import gc
import gzip
import hashlib
import json
import pickle
import random
import re
import xml.etree.ElementTree as ET

from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from sklearn.utils.extmath import randomized_svd
from lhotse import (
    CutSet,
    RecordingSet,
    SupervisionSet,
)
from lhotse.features import (
    Mfcc,
    MfccConfig,
)

from tqdm import tqdm

from config import MANIFEST_PATH




##### Lhotse #####
def get_cuts(corpus_name: str, partition_names: list[str], manifest_dir: str = "."):
    cuts = {}

    for partition in partition_names:
        recordings = RecordingSet.from_file(
            f"{manifest_dir}/{corpus_name}/{partition}_recording_set.jsonl.gz"
        )
        supervisions = SupervisionSet.from_file(
            f"{manifest_dir}/{corpus_name}/{partition}_supervision_set.jsonl.gz"
        )

        cuts[partition] = CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )

    return cuts



##### General #####
class PhoneClass(Enum):
    MONOPHONE = "monophone"
    DIPHONE = "diphone"
    TRIPHONE = "triphone"


def write_json(file_path: str, data: dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""
    with Path(file_path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_json(file_path: str) -> dict[str, Any]:
    """Read a JSON file and return its contents."""
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path):
    path = Path(path)

    with open(path, "rb") as f:
        fixed_selection = pickle.load(f)

    return fixed_selection


def load_lexicon(path):
    with gzip.open(path, "rb") as f:
        root = ET.parse(f).getroot()

    return {
        lemma.findtext("orth").strip().upper(): lemma.findtext("phon").strip().split()
        for lemma in root.findall(".//lemma")
        if lemma.findtext("orth") and lemma.findtext("phon")
    }


def load_feature_pickles(feature_dir: str | Path) -> dict:
    """Load and merge all feature pickle files in a directory."""
    feature_dir = Path(feature_dir)
    features = {}

    for path in sorted(feature_dir.glob("*.pickle")):
        print(f"Loading {path}")
        with path.open("rb") as f:
            features.update(pickle.load(f))

    return features


def normalize_phone_label(
    phone: str | tuple[str, str],
) -> str | tuple[str, str]:
    """
    Normalize a phone or diphone label for dictionary lookup.

    Single phones are converted to uppercase strings. Diphones are
    converted to tuples containing uppercase phone labels.
    """
    if isinstance(phone, tuple):
        return tuple(element.upper() for element in phone)

    return phone.upper()


def print_distance_summary(
    final_distances: Dict[str, Dict[str, List[float]]],
) -> None:
    """
    Print mean and standard deviation of distances for each metric
    and level.

    Parameters
    ----------
    final_distances
        Dictionary mapping levels to dictionaries of metrics. Each
        metric maps to a list of distance values.
    """
    if not final_distances:
        return

    for metric in next(iter(final_distances.values())):
        print(f"\n{'=' * 65}")
        print(f"{metric:^65}")
        print(f"{'=' * 65}")
        print(f"{'Level':<10}" f"{'N':>8}" f"{'Mean':>15}" f"{'Std':>15}")
        print("-" * 65)

        for level, level_distances in final_distances.items():
            values = level_distances.get(metric, [])

            if values:
                print(
                    f"{level:<10}"
                    f"{len(values):>8}"
                    f"{np.mean(values):>15.3f}"
                    f"{np.std(values):>15.3f}"
                )
            else:
                print(f"{level:<10}" f"{0:>8}" f"{'NaN':>15}" f"{'NaN':>15}")



def print_distance_summary_by_part(
    final_distances: dict[str, dict[str, dict[str, list[float]]]],
    metrics: list[str] | None = None,
) -> None:
    if not final_distances:
        return

    parts = list(final_distances)
    available_metrics = next(iter(final_distances[parts[0]].values()))

    if metrics is None:
        metrics = list(available_metrics)

    levels = list(final_distances[parts[0]])

    for metric in metrics:
        width = 25 + 18 * len(parts)
        print(f"\n{'=' * width}\n{metric:^{width}}\n{'=' * width}")
        print(f"{'Level':<10}{'N':>8}" + "".join(f"{p:>18}" for p in parts))
        print("-" * width)

        for level in levels:
            row = f"{level:<10}"
            values_by_part = [
                final_distances[p].get(level, {}).get(metric, [])
                for p in parts
            ]
            row += f"{sum(map(len, values_by_part)):>8}"

            for values in values_by_part:
                row += (
                    f"{np.mean(values):.3f} ({np.std(values):.3f})".rjust(18)
                    if values
                    else f"{'NaN':>18}"
                )

            print(row)




def intersect_nested_dict_keys(
    data: dict[str, dict[str, Any]],
) -> set[str]:
    """Return keys present in every nested dictionary.

    Args:
        data: A dictionary whose values are nested dictionaries.

    Returns:
        The set of keys shared by all nested dictionaries.
        Returns an empty set if `data` is empty.
    """
    if not data:
        return set()

    return set.intersection(*(set(d.keys()) for d in data.values()))



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
##### Math #####
def project_to_svd_basis(X_ref, X_test, rank=10):
    U, sigma, VT = randomized_svd(X_ref, n_components=rank)
    C_ref = U * sigma
    C_test = X_test @ np.linalg.pinv(VT)

    return C_ref, C_test, (U, sigma, VT)


def get_transformed(X_test, VT_ref):
    return X_test @ np.linalg.pinv(VT_ref)


def subtract_mean(X):
    mean_X = np.mean(X, axis=0, keepdims=True)

    return X - mean_X


def mean_normalize_wavlm_features(
    features_by_cut: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Create a mean-normalized copy of WavLM features for each cut.

    Each utterance is normalized independently by subtracting its mean
    feature vector across frames.

    Parameters
    ----------
    features_by_cut
        Dictionary mapping cutset indices to WavLM feature matrices
        of shape (T, D).

    Returns
    -------
    Dict[int, np.ndarray]
        New dictionary with the same keys and mean-normalized features.
    """
    return {
        cut_index: features - features.mean(axis=0, keepdims=True)
        for cut_index, features in features_by_cut.items()
    }


##### Feature extraction #####



def extract_features(
    cut: Any,
    model: torch.nn.Module,
    device: str = "cuda",
    layer: int = 6,
) -> np.ndarray:
    """
    Extract features from any Hugging Face WavLM / wav2vec2 model.

    Works for:
        - microsoft/wavlm-base-plus
        - microsoft/wavlm-large
        - facebook/wav2vec2-large-lv60
        - facebook/wav2vec2-large-xlsr-53

    Assumes cut.load_audio() returns shape (1, n_samples).
    """

    wav = torch.as_tensor(
        cut.load_audio()[:, :],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(
            input_values=wav,
            output_hidden_states=True,
            return_dict=True,
        )

    features = outputs.hidden_states[layer]

    return features.squeeze(0).cpu().numpy()

def extract_and_save_features(
    cuts: Iterable[Any],
    model: torch.nn.Module,
    split: str,
    output_dir: str | Path,
    batch_size: int = 1000,
    device: str = "cuda",
    layer: int = 6,
    indices: Optional[Collection[int]] = None,
) -> None:
    """Extract features and save batches as {recording_id: feature} dictionaries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.eval()

    allowed_indices = None if indices is None else set(indices)

    batch: dict[str, np.ndarray] = {}
    batch_start = 0
    num_processed = 0
    num_skipped = 0

    for idx, cut in enumerate(cuts):
        if allowed_indices is not None and idx not in allowed_indices:
            num_skipped += 1
            continue

        recording_id = cut.recording_id

        features = extract_features(
            cut=cut,
            model=model,
            device=device,
            layer=layer,
        )

        batch[recording_id] = features
        num_processed += 1
        del features

        if len(batch) >= batch_size:
            batch_end = batch_start + len(batch)
            filename = output_path / f"{split}_{batch_start}-{batch_end}.pickle"

            with filename.open("wb") as f:
                pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Saved {filename}")

            batch.clear()
            batch_start = batch_end

            gc.collect()

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if batch:
        batch_end = batch_start + len(batch)
        filename = output_path / f"{split}_{batch_start}-{batch_end}.pickle"

        with filename.open("wb") as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {filename}")

        batch.clear()
        gc.collect()

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Done. Processed: {num_processed}, skipped: {num_skipped}")



def extract_and_save_mfcc(
    cuts: Iterable[Any],
    split: str,
    output_dir: str | Path,
    batch_size: int = 1000,
    num_ceps: int = 13,
    frame_shift: float = 0.01,
    frame_length: float = 0.025,
    high_freq: float | None = None,
    subsampling: int = 1,
    indices: Collection[int] | None = None,
) -> None:
    """Extract MFCCs and average-pool consecutive frames."""
    if subsampling < 1:
        raise ValueError(f"subsampling must be >= 1, got {subsampling}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    allowed_indices = None if indices is None else set(indices)

    config = MfccConfig(
        num_ceps=num_ceps,
        frame_shift=frame_shift,
        frame_length=frame_length,
        high_freq=high_freq,
    )
    extractor = Mfcc(config)

    batch: dict[str, np.ndarray] = {}
    batch_start = 0
    num_processed = 0
    num_skipped = 0

    for idx, cut in enumerate(cuts):
        if allowed_indices is not None and idx not in allowed_indices:
            num_skipped += 1
            continue

        features = cut.compute_features(extractor=extractor)

        if subsampling > 1:
            n_frames = features.shape[0] // subsampling
            features = features[:n_frames * subsampling]
            features = features.reshape(
                n_frames, subsampling, features.shape[-1]
            ).mean(axis=1)

        batch[cut.recording_id] = features
        num_processed += 1

        if len(batch) >= batch_size:
            batch_end = batch_start + len(batch)
            filename = output_path / f"{split}_{batch_start}-{batch_end}.pickle"

            with filename.open("wb") as f:
                pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Saved {filename}")

            batch.clear()
            batch_start = batch_end
            gc.collect()

    if batch:
        batch_end = batch_start + len(batch)
        filename = output_path / f"{split}_{batch_start}-{batch_end}.pickle"

        with filename.open("wb") as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {filename}")

    print(f"Done. Processed: {num_processed}, skipped: {num_skipped}")

###### Data preparation ######
def convert_ctc_alignment_to_spans(alignment):
    """Convert (end_frame, token_idx, phone) -> (start_frame, end_frame, phone)."""
    spans = []
    start = 0

    for end, _, phone in alignment:
        end = int(end)

        if end <= start:
            continue

        spans.append((start, end, str(phone)))
        start = end

    return spans

def collect_all_speakers(
        datasets,
        sentence=None,
        by_recording_id=False,
):
    """
    Args:
        datasets: List of (cuts, features) tuples.
        sentence: str can be for example SA1

    Returns:
        {
            "male": {speaker_id: info, ...},
            "female": {speaker_id: info, ...},
        }
    """

    speakers = {"male": {}, "female": {}}

    def get_region(cut):
        return next(
            p for p in Path(cut.recording.sources[0].source).parts
            if p.startswith("DR")
        )

    for cuts, features in datasets:
        for idx, cut in enumerate(cuts):

            if sentence is not None and f"-{sentence}-" not in cut.id:
                continue

            sup = cut.supervisions[0]

            feature_idx = cut.recording_id if by_recording_id else idx

            speakers[sup.gender][sup.speaker] = {
                "speaker": sup.speaker,
                "gender": sup.gender,
                "region": get_region(cut),
                "cut": cut,
                "features": features[feature_idx],
            }

    return speakers


def convert_feature_pickles_to_recording_id_subsets(
    feature_path: str | Path,
    cuts: Any,
    global_indices: list[int] | None = None,
    output_filename: str | Path | None = None,
    output_prefix: str = "recording_id_",
) -> None:
    """
    Convert chunked feature pickles from global-index ordering to
    recording-ID keyed pickles.

    When ``global_indices`` is provided, only those global indices are
    selected and all selected features are written into one pickle.
    """
    feature_path = Path(feature_path)

    pickle_files = sorted(
        feature_path.glob("*.pickle"),
        key=lambda path: int(
            path.stem.rsplit("_", 1)[1].split("-")[0]
        ),
    )

    if global_indices is not None:
        if output_filename is None:
            raise ValueError(
                "output_filename must be provided when "
                "global_indices is provided."
            )

        selected_indices = set(global_indices)
        converted_features: dict[str, Any] = {}

        # Keep the global indices in their original order.
        for global_index in tqdm(
            global_indices,
            desc="Converting global indices",
        ):
            # Find the pickle containing this global index.
            for pickle_file in pickle_files:
                range_part = pickle_file.stem.rsplit("_", 1)[1]

                start_index, end_index = map(
                    int,
                    range_part.split("-"),
                )

                if not (start_index <= global_index < end_index):
                    continue

                with pickle_file.open("rb") as file:
                    features = pickle.load(file)

                local_index = global_index - start_index
                feature = features[local_index]

                recording_id = str(
                    cuts[global_index].recording_id
                )

                if recording_id in converted_features:
                    raise ValueError(
                        f"Duplicate recording ID {recording_id!r} "
                        f"at global index {global_index}."
                    )

                converted_features[recording_id] = feature

                del features
                break
            else:
                raise ValueError(
                    f"Global index {global_index} was not found "
                    f"in any feature pickle."
                )

        output_file = Path(output_filename)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open("wb") as file:
            pickle.dump(
                converted_features,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"Saved {len(converted_features)} entries to "
            f"{output_file}"
        )

        return

    # Original full conversion mode.
    for pickle_file in tqdm(
        pickle_files,
        desc="Converting pickle chunks",
    ):
        range_part = pickle_file.stem.rsplit("_", 1)[1]

        start_index, end_index = map(
            int,
            range_part.split("-"),
        )

        with pickle_file.open("rb") as file:
            features = pickle.load(file)

        expected_size = end_index - start_index

        assert len(features) == expected_size, (
            f"{pickle_file}: expected {expected_size} features, "
            f"got {len(features)}"
        )

        converted_features: dict[str, Any] = {}

        for local_index, feature in features.items():
            global_index = start_index + local_index

            recording_id = str(
                cuts[global_index].recording_id
            )

            if recording_id in converted_features:
                raise ValueError(
                    f"Duplicate recording ID {recording_id!r} "
                    f"in {pickle_file.name}."
                )

            converted_features[recording_id] = feature

        output_file = feature_path / (
            f"{output_prefix}{pickle_file.name}"
        )

        with output_file.open("wb") as file:
            pickle.dump(
                converted_features,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        del features
        del converted_features



###Data Selecetion#########
def select_diverse_cuts(
    cuts: CutSet,
    target_hours: float = 4.0,
    max_repeats_per_text: int = 2,
    seed: int = 42,
):
    """Fast speaker/text-diverse selection from a Lhotse CutSet."""

    def norm(text):
        return re.sub(
            r"\s+", " ",
            re.sub(r"[^\w\s]", "", (text or "").lower())
        ).strip()

    rng = random.Random(seed)
    target = target_hours * 3600

    # Materialize once.
    cuts = list(cuts)

    # text -> cuts, speaker -> cuts
    by_text = defaultdict(list)
    by_speaker = defaultdict(list)

    for cut in cuts:
        if not cut.supervisions:
            continue

        sup = cut.supervisions[0]
        spk = (sup.custom or {}).get("speaker", {})
        spk = spk.get("id") if isinstance(spk, dict) else spk
        text = norm(sup.text)

        if spk and text:
            by_text[text].append(cut)
            by_speaker[spk].append(cut)

    # Shuffle once.
    for xs in by_text.values():
        rng.shuffle(xs)
    for xs in by_speaker.values():
        rng.shuffle(xs)

    selected = []
    used_text = defaultdict(int)
    used_speaker = defaultdict(float)
    selected_ids = set()
    duration = 0.0

    # ------------------------------------------------------------
    # 1. One short representative per speaker.
    # ------------------------------------------------------------
    reps = []

    for spk, xs in by_speaker.items():
        valid = [
            c for c in xs
            if used_text[norm(c.supervisions[0].text)] < max_repeats_per_text
        ]

        if valid:
            # Short cuts maximize number of speakers fitting in 4h.
            c = min(valid, key=lambda x: x.duration)
            reps.append((spk, c))

    rng.shuffle(reps)

    for spk, c in reps:
        if duration + c.duration > target:
            continue

        text = norm(c.supervisions[0].text)
        selected.append(c)
        selected_ids.add(c.id)
        used_text[text] += 1
        used_speaker[spk] += c.duration
        duration += c.duration

    # ------------------------------------------------------------
    # 2. Fast global fill.
    #
    # Build a candidate pool ONCE instead of rescanning everything.
    # ------------------------------------------------------------
    pool = []

    for text, xs in by_text.items():
        # At most max_repeats_per_text candidates for each text.
        for c in xs[:max_repeats_per_text]:
            if c.id not in selected_ids:
                sup = c.supervisions[0]
                spk = sup.custom["speaker"]["id"]
                pool.append((spk, text, c))

    rng.shuffle(pool)

    # Prefer less represented speakers and unseen texts.
    # Sort only once.
    pool.sort(
        key=lambda x: (
            used_speaker[x[0]],
            used_text[x[1]],
            x[2].duration,
        )
    )

    for spk, text, c in pool:
        if duration >= target:
            break

        if c.id in selected_ids:
            continue

        if used_text[text] >= max_repeats_per_text:
            continue

        if duration + c.duration > target:
            continue

        selected.append(c)
        selected_ids.add(c.id)
        used_text[text] += 1
        used_speaker[spk] += c.duration
        duration += c.duration

    result = CutSet.from_cuts(selected)

    return result, duration, len(used_speaker), dict(used_speaker)


## RASR #####
PhoneInterval = tuple[int, int, str]
def build_wavlm_phone_intervals(
    alignment: Sequence[str],
    alignment_frame_duration_ms: int = 40,
    wavlm_frame_duration_ms: int = 20,
) -> list[PhoneInterval]:
    """Convert a frame-level phone alignment to WavLM frame intervals.

    Consecutive alignment labels with the same center phone are merged.
    The resulting boundaries are expressed as WavLM frame indices rather
    than seconds.

    Parameters
    ----------
    alignment
        Sequence of alignment labels, e.g.
        ``"AY{L+SH}@i@f.0"`` or
        ``"[SILENCE]{#+#}@i@f.0"``.

    alignment_frame_duration_ms
        Duration represented by each alignment label in milliseconds.
        For the described alignment this is 40 ms.

    wavlm_frame_duration_ms
        Temporal resolution of the WavLM representations in milliseconds.
        For the described representations this is 20 ms.

    Returns
    -------
    list[PhoneInterval]
        A list of ``(start_frame, end_frame, center_phone)`` tuples,
        where the frame indices refer to WavLM frames and ``end_frame``
        is exclusive.

        Silence is represented as ``"SIL"``.

    Examples
    --------
    If the alignment is::

        [
            "[SILENCE]{#+#}@i@f.0",
            "[SILENCE]{#+#}@i@f.0",
            "AY{#+TH}@i@f.0",
            "AY{#+TH}@i@f.0",
            "TH{AY+IH}@i.0",
        ]

    the result is::

        [
            (0, 4, "sil"),
            (4, 8, "AY"),
            (8, 10, "TH"),
        ]
    """
    if not alignment:
        return []

    if alignment_frame_duration_ms <= 0:
        raise ValueError(
            "alignment_frame_duration_ms must be positive."
        )

    if wavlm_frame_duration_ms <= 0:
        raise ValueError(
            "wavlm_frame_duration_ms must be positive."
        )

    if alignment_frame_duration_ms % wavlm_frame_duration_ms != 0:
        raise ValueError(
            "alignment_frame_duration_ms must be an integer multiple "
            "of wavlm_frame_duration_ms."
        )

    wavlm_frames_per_alignment_frame = (
        alignment_frame_duration_ms // wavlm_frame_duration_ms
    )

    def get_center_phone(label: str) -> str:
        """Extract the center phone from an alignment label."""
        center_phone = label.split("{", 1)[0]

        if center_phone == "[SILENCE]":
            return "SIL"

        return center_phone

    intervals: list[PhoneInterval] = []

    current_phone = get_center_phone(alignment[0])
    start_frame = 0

    for alignment_index, label in enumerate(alignment[1:], start=1):
        phone = get_center_phone(label)

        if phone != current_phone:
            end_frame = (
                alignment_index
                * wavlm_frames_per_alignment_frame
            )

            intervals.append(
                (start_frame, end_frame, current_phone)
            )

            current_phone = phone
            start_frame = end_frame

    end_frame = (
        len(alignment)
        * wavlm_frames_per_alignment_frame
    )

    intervals.append(
        (start_frame, end_frame, current_phone)
    )

    return intervals



def select_cuts_up_to_hours(
    cuts,
    max_hours: float = 4.0,
    seed: int | None = 42,
) -> tuple[list[int], dict[int, str]]:
    """Randomly select cuts up to max_hours and return their speaker identities."""
    rng = random.Random(seed)
    max_duration = max_hours * 3600

    speaker_cuts = defaultdict(list)
    for idx, cut in enumerate(cuts):
        speaker_cuts[cut.supervisions[0].speaker].append(idx)

    speakers = list(speaker_cuts)
    rng.shuffle(speakers)

    for speaker in speakers:
        rng.shuffle(speaker_cuts[speaker])

    selected_indices = []
    total_duration = 0.0

    while total_duration < max_duration:
        added = False

        for speaker in speakers:
            if not speaker_cuts[speaker]:
                continue

            idx = speaker_cuts[speaker].pop()
            duration = cuts[idx].duration

            if total_duration + duration <= max_duration:
                selected_indices.append(idx)
                total_duration += duration
                added = True

        if not added:
            break

    rng.shuffle(selected_indices)

    selected_speakers = {
        idx: cuts[idx].supervisions[0].speaker
        for idx in selected_indices
    }

    print(
        f"Selected {len(selected_indices)} cuts "
        f"({total_duration / 3600:.2f} hours) "
        f"from {len(set(selected_speakers.values()))} speakers"
    )

    return selected_indices, selected_speakers



#For white noise injected to cosine distance choose different seeds for each recording
def stable_seed(
    base_seed: int,
    *items: Any,
) -> int:
    text = "|".join([str(base_seed)] + [str(x) for x in items])
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32)


def make_cosine_noise_suffix(
    params: dict[str, Any] | None,
) -> str:
    """
    Create filename suffix for cosine-noise experiments.
    Must match the saving code.
    """
    if params is None:
        return ""

    mode = params.get("cosine_noise_mode", "none")

    if mode == "none":
        return ""

    level = params.get("cosine_noise_level", 1.0)
    seed = params.get("cosine_noise_seed", None)

    return f"_cosnoise-{mode}_lvl{level}_seed{seed}"