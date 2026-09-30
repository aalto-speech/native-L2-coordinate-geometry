from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, TypeAlias

import gc
import pickle
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd

from utils import (
    collect_all_speakers,
    mean_normalize_wavlm_features,
    PhoneClass,
)

from IPython import embed




class AveragingStrategy(Enum):
    CENTER_PHONE = "center_phone"
    FULL_UNIT = "full_unit"


FeatureVector: TypeAlias = NDArray
FeatureMatrix: TypeAlias = NDArray
PhoneLabel: TypeAlias = str | tuple[str, ...]
FrameSpan: TypeAlias = tuple[int, int]




SIL_LABEL = "SIL"

SILENCE_SYMBOLS = frozenset(
    {
        "",
        "#",
        "h#",
        "sil",
        "pau",
        "epi",
        "SIL",
    }
)

WORD_BOUNDARY_SYMBOLS = frozenset(
    {
        "|",
    }
)


@dataclass
class PhoneSegment:
    symbol: str
    spans: tuple[FrameSpan, ...]
    is_silence: bool

    @property
    def start(self) -> int:
        return self.spans[0][0]

    @property
    def end(self) -> int:
        return self.spans[-1][1]


@dataclass
class PhoneUnit:
    label: PhoneLabel
    spans: tuple[FrameSpan, ...]


# ---------------------------------------------------------------------
# Public function for TIMIT-style dataset
# ---------------------------------------------------------------------
def map_phone_segments(
    segments: list[PhoneSegment],
    phone_mapping: dict[str, str | None],
) -> list[PhoneSegment]:
    """
    Map phone symbols using the supplied phone mapping.

    Phones mapped to ``None`` are removed.

    Parameters
    ----------
    segments:
        Normalized phone segments.

    phone_mapping:
        Mapping from source phone symbols to canonical phone symbols.
        A value of ``None`` means that the phone should be discarded.

    Returns
    -------
    list[PhoneSegment]
        Phone segments with mapped symbols.
    """
    mapped_segments: list[PhoneSegment] = []

    for segment in segments:
        mapped_symbol = phone_mapping.get(
            segment.symbol,
            segment.symbol,
        )

        if mapped_symbol is None:
            continue

        mapped_segments.append(
            PhoneSegment(
                symbol=mapped_symbol,
                spans=segment.spans,
                is_silence=segment.is_silence,
            )
        )

    return mapped_segments

def build_average_phone_matrix_for_timit(
    dataset: dict,
    phone_class: PhoneClass | str,
    strategy: AveragingStrategy | str,
    phone_mapping: dict[str, str | None] | None = None,
    by_gender: bool = False,
    frame_shift: float = 0.02,
    collapse_repeats: bool = True,
) -> (
    tuple[FeatureMatrix, list[PhoneLabel]]
    | dict[str, tuple[FeatureMatrix, list[PhoneLabel]]]
):
    """
    Build average monophone/diphone/triphone codebooks from a TIMIT-style
    dataset.

    This function expects the original TIMIT-like structure:

        dataset["male"][utt_id]["features"]
        dataset["male"][utt_id]["cut"].supervisions[0].alignment["phone"]

    and similarly for "female".

    Parameters
    ----------
    dataset:
        TIMIT-style dataset dictionary.

    phone_class:
        "monophone", "diphone", "triphone" or PhoneClass enum.

    strategy:
        "center_phone", "full_unit" or AveragingStrategy enum.

    by_gender:
        If True, return separate male/female codebooks.

    frame_shift:
        Duration of one feature frame in seconds.

    collapse_repeats:
        If True, consecutive identical phone symbols are collapsed.
    """

    phone_class = _coerce_phone_class(phone_class)
    strategy = _coerce_strategy(strategy)

    groups = ("male", "female") if by_gender else ("all",)
    result: dict[str, tuple[FeatureMatrix, list[PhoneLabel]]] = {}

    for group in groups:
        genders = (group,) if group != "all" else ("male", "female")

        items = []

        for gender in genders:
            for info in dataset.get(gender, {}).values():
                features = np.asarray(info["features"])
                timit_alignment = info["cut"].supervisions[0].alignment["phone"]


                segments = segments_from_timit_alignment(
                    timit_alignment,
                    collapse_repeats=collapse_repeats,
                    frame_shift=frame_shift,
                )

                if phone_mapping is not None:
                    segments = map_phone_segments(
                        segments,
                        phone_mapping,
                    )

                items.append((features, segments))


        result[group] = build_average_phone_matrix_from_segments(
            items=items,
            phone_class=phone_class,
            strategy=strategy,

        )

    return result if by_gender else result["all"]


# ---------------------------------------------------------------------
# Public function for tuple alignments
# ---------------------------------------------------------------------


def build_average_phone_matrix_from_tuple_alignments(
    items: Iterable[
        tuple[FeatureMatrix, Iterable[tuple[int, int, str]]]
    ],
    phone_class: PhoneClass | str,
    strategy: AveragingStrategy | str,
    collapse_repeats: bool = False,
) -> tuple[FeatureMatrix, list[PhoneLabel]]:
    """
    Build average monophone, diphone, or triphone vectors from WavLM
    frame-index alignments.

    Each alignment tuple is:

        (start_frame, end_frame, phone)

    The frame indices directly index the corresponding WavLM feature
    matrix.
    """
    phone_class = _coerce_phone_class(phone_class)
    strategy = _coerce_strategy(strategy)

    normalized_items: list[
        tuple[FeatureMatrix, list[PhoneSegment]]
    ] = []

    for features, tuple_alignment in items:
        features = np.asarray(features)

        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape (T, D), "
                f"got {features.shape}"
            )

        segments = segments_from_tuple_alignment(
            tuple_alignment,
            collapse_repeats=collapse_repeats,
        )

        normalized_items.append((features, segments))

    return build_average_phone_matrix_from_segments(
        items=normalized_items,
        phone_class=phone_class,
        strategy=strategy,
    )


# ---------------------------------------------------------------------
# Shared core averaging function
# ---------------------------------------------------------------------


def build_average_phone_matrix_from_segments(
    items: Iterable[tuple[FeatureMatrix, list[PhoneSegment]]],
    phone_class: PhoneClass | str,
    strategy: AveragingStrategy | str,
) -> tuple[FeatureMatrix, list[PhoneLabel]]:
    """
    Build average phone vectors from normalized frame-index segments.
    """
    phone_class = _coerce_phone_class(phone_class)
    strategy = _coerce_strategy(strategy)

    label_vectors: defaultdict[
        PhoneLabel,
        list[FeatureVector],
    ] = defaultdict(list)

    feature_dim: int | None = None

    for features, segments in items:
        features = np.asarray(features)

        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape (T, D), "
                f"got {features.shape}"
            )

        if feature_dim is None:
            feature_dim = features.shape[1]

        if not segments:
            continue

        units = _build_phone_units(
            segments=segments,
            phone_class=phone_class,
            strategy=strategy,
        )

        for unit in units:
            vector = _average_feature_spans(
                features=features,
                spans=unit.spans,
            )

            if vector is not None:
                label_vectors[unit.label].append(vector)

    labels = sorted(
        label_vectors.keys(),
        key=_label_sort_key,
    )

    if labels:
        matrix = np.stack(
            [
                np.mean(label_vectors[label], axis=0)
                for label in labels
            ],
            axis=0,
        )
    else:
        matrix = np.empty(
            (0, feature_dim or 0),
            dtype=np.float64,
        )

    return matrix, labels


# ---------------------------------------------------------------------
# Alignment adapters
# ---------------------------------------------------------------------
def _merge_timit_closures(
    raw_spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """
    Merge TIMIT closure phones into adjacent lexical phones.

    ``cl`` and ``vcl`` represent closure portions of stop/affricate
    realizations. They are not present in the LibriSpeech-style phone
    inventory used by the downstream alignments.

    The closure interval is therefore absorbed by the preceding phone
    when possible, otherwise by the following phone.

    The original frame boundaries are preserved exactly. No new
    timestamp quantization is performed here.
    """

    closure_symbols = {"cl", "vcl"}

    spans = list(raw_spans)
    result: list[tuple[int, int, str]] = []

    i = 0

    while i < len(spans):
        start, end, symbol = spans[i]

        if symbol not in closure_symbols:
            result.append((start, end, symbol))
            i += 1
            continue

        # Prefer the preceding non-closure phone.
        if result and result[-1][2] not in closure_symbols:
            prev_start, _, prev_symbol = result[-1]
            result[-1] = (
                prev_start,
                end,
                prev_symbol,
            )
            i += 1
            continue

        # Otherwise attach to the following non-closure phone.
        if i + 1 < len(spans):
            next_start, next_end, next_symbol = spans[i + 1]

            if next_symbol not in closure_symbols:
                result.append(
                    (
                        start,
                        next_end,
                        next_symbol,
                    )
                )
                i += 2
                continue

        # Degenerate case: closure has no usable neighbor.
        i += 1

    return result

def segments_from_timit_alignment(
    alignment: Iterable[Any],
    frame_shift: float = 0.02,
    collapse_repeats: bool = True,
) -> list[PhoneSegment]:
    """
    Convert TIMIT/Lhotse time-based phone alignments into WavLM
    frame-indexed PhoneSegment objects.

    TIMIT phone timestamps are in seconds. Frame boundaries are obtained
    from the timestamps using a single global time-to-frame mapping.

    Closure phones (``cl`` and ``vcl``) are not retained as independent
    phone labels. Their time intervals are merged into the neighboring
    non-closure phone so that their frames are not discarded.

    Silence phones are mapped to ``SIL``.
    """

    if frame_shift <= 0:
        raise ValueError(
            f"frame_shift must be positive, got {frame_shift}"
        )

    raw_spans: list[tuple[int, int, str]] = []

    for phone in alignment:
        start_seconds = float(phone.start)
        end_seconds = float(phone.start + phone.duration)
        symbol = str(phone.symbol).strip().lower()

        start_frame = int(
            np.floor(start_seconds / frame_shift)
        )
        end_frame = int(
            np.ceil(end_seconds / frame_shift)
        )

        if end_frame <= start_frame:
            continue

        raw_spans.append(
            (start_frame, end_frame, symbol)
        )

    raw_spans = _merge_timit_closures(raw_spans)

    return _segments_from_raw_spans(
        raw_spans=raw_spans,
        collapse_repeats=collapse_repeats,
    )


def segments_from_tuple_alignment(
    alignment: Iterable[tuple[int, int, str]],
    collapse_repeats: bool = False,
) -> list[PhoneSegment]:
    """
    Convert WavLM frame-index alignments into PhoneSegment objects.

    Each tuple is:

        (start_frame, end_frame, phone)

    where the frame indices directly index the WavLM feature matrix.
    """
    raw_spans = [
        (int(start), int(end), str(symbol))
        for start, end, symbol in alignment
    ]

    return _segments_from_raw_spans(
        raw_spans=raw_spans,
        collapse_repeats=collapse_repeats,
    )


def _segments_from_raw_spans(
    raw_spans: Iterable[tuple[int, int, str]],
    collapse_repeats: bool,
) -> list[PhoneSegment]:
    """
    Normalize frame-index alignment spans.

    Word-boundary symbols are removed, silence symbols are mapped to
    ``SIL``, and consecutive identical symbols can optionally be merged.
    """
    segments: list[PhoneSegment] = []

    for start, end, raw_symbol in raw_spans:
        raw_symbol = str(raw_symbol).strip()

        if _is_word_boundary_symbol(raw_symbol):
            continue

        if end <= start:
            continue

        symbol, is_silence = _canonical_symbol(raw_symbol)

        current = PhoneSegment(
            symbol=symbol,
            spans=((int(start), int(end)),),
            is_silence=is_silence,
        )

        if (
            collapse_repeats
            and segments
            and segments[-1].symbol == current.symbol
            and segments[-1].is_silence == current.is_silence
        ):
            previous = segments[-1]
            previous.spans = previous.spans + current.spans
        else:
            segments.append(current)

    return segments


# ---------------------------------------------------------------------
# Shared phone-unit logic
# ---------------------------------------------------------------------


def _build_phone_units(
    segments: list[PhoneSegment],
    phone_class: PhoneClass,
    strategy: AveragingStrategy,
) -> list[PhoneUnit]:
    """
    Build monophone, diphone, or triphone units.

    phone_class determines the label identity.

    strategy determines which frames are averaged.

    The center phone is always the current segment.

    Monophone label:
        center

    Diphone label:
        left, center

    Triphone label:
        left, center, right

    Universal silence rule:
        If the center phone is SIL, skip the unit.

    Silence may appear as context:

        ("SIL", "AA")
        ("SIL", "AA", "B")
        ("AA", "B", "SIL")

    But the following are skipped:

        ("AA", "SIL")
        ("AA", "SIL", "B")
    """

    units: list[PhoneUnit] = []

    for i, center in enumerate(segments):
        left = segments[i - 1] if i > 0 else None
        right = segments[i + 1] if i < len(segments) - 1 else None

        # Critical rule:
        # skip only if the center phone is silence.
        if center.is_silence:
            continue

        if phone_class is PhoneClass.MONOPHONE:
            label: PhoneLabel = center.symbol

            # For monophones, both strategies are equivalent.
            spans = center.spans

        elif phone_class is PhoneClass.DIPHONE:
            left_symbol = left.symbol if left is not None else SIL_LABEL

            label = (
                left_symbol,
                center.symbol,
            )

            if strategy is AveragingStrategy.CENTER_PHONE:
                # Diphone label, but average center phone only.
                spans = center.spans

            elif strategy is AveragingStrategy.FULL_UNIT:
                # Diphone label, average left + center.
                if left is not None:
                    spans = left.spans + center.spans
                else:
                    spans = center.spans

            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

        elif phone_class is PhoneClass.TRIPHONE:
            left_symbol = left.symbol if left is not None else SIL_LABEL
            right_symbol = right.symbol if right is not None else SIL_LABEL

            label = (
                left_symbol,
                center.symbol,
                right_symbol,
            )

            if strategy is AveragingStrategy.CENTER_PHONE:
                # Triphone label, but average center phone only.
                spans = center.spans

            elif strategy is AveragingStrategy.FULL_UNIT:
                # Triphone label, average left + center + right.
                spans_list: list[TimeSpan] = []

                if left is not None:
                    spans_list.extend(left.spans)

                spans_list.extend(center.spans)

                if right is not None:
                    spans_list.extend(right.spans)

                spans = tuple(spans_list)

            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

        else:
            raise ValueError(f"Unsupported phone_class: {phone_class}")

        units.append(
            PhoneUnit(
                label=label,
                spans=spans,
            )
        )

    return units


# ---------------------------------------------------------------------
# Feature averaging
# ---------------------------------------------------------------------


def _average_feature_spans(
    features: FeatureMatrix,
    spans: tuple[FrameSpan, ...],
) -> FeatureVector | None:
    """
    Average WavLM feature frames over one or more frame-index spans.

    Each span is interpreted directly as a Python slice:

        (start_frame, end_frame) -> features[start_frame:end_frame]
    """
    chunks: list[FeatureMatrix] = []

    for start_frame, end_frame in spans:
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, features.shape[0])

        if end_frame > start_frame:
            chunks.append(features[start_frame:end_frame])

    if not chunks:
        return None

    return np.concatenate(chunks, axis=0).mean(axis=0)


# ---------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------


def _is_word_boundary_symbol(symbol: str) -> bool:
    return str(symbol).strip() in WORD_BOUNDARY_SYMBOLS


def _is_silence_symbol(symbol: str) -> bool:
    normalized = str(symbol).strip().lower()

    return normalized in {
        silence_symbol.lower()
        for silence_symbol in SILENCE_SYMBOLS
    }


def _canonical_symbol(symbol: str) -> tuple[str, bool]:
    """
    Convert raw symbols to canonical labels.

    Silence/boundary phones are mapped to "SIL".

    Word boundary "|" should already have been removed before this point.
    """

    symbol = str(symbol).strip()

    if _is_silence_symbol(symbol):
        return SIL_LABEL, True

    return symbol, False


def _coerce_phone_class(phone_class: PhoneClass | str) -> PhoneClass:
    if isinstance(phone_class, PhoneClass):
        return phone_class

    try:
        return PhoneClass(phone_class)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in PhoneClass)
        raise ValueError(
            f"Unsupported phone_class {phone_class!r}. "
            f"Allowed values are: {allowed}"
        ) from exc


def _coerce_strategy(strategy: AveragingStrategy | str) -> AveragingStrategy:
    if isinstance(strategy, AveragingStrategy):
        return strategy

    try:
        return AveragingStrategy(strategy)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AveragingStrategy)
        raise ValueError(
            f"Unsupported strategy {strategy!r}. "
            f"Allowed values are: {allowed}"
        ) from exc


def _label_sort_key(label: PhoneLabel) -> tuple:
    if isinstance(label, str):
        return (1, label)

    return (len(label), *label)



#-------------------------------------------------------------
#Building the averages
#-------------------------------------------------------------
def build_phone_codebook_timit(
    cuts_train_timit: Any,
    wavlm_feat_train_timit: dict[int, np.ndarray],
    cuts_dev_timit: Any,
    wavlm_feat_dev_timit: dict[int, np.ndarray],
    cuts_test_timit: Any,
    wavlm_feat_test_timit: dict[int, np.ndarray],
    rank: int,
    phone_class: PhoneClass = PhoneClass.DIPHONE,
    strategy: AveragingStrategy = AveragingStrategy.CENTER_PHONE,
    phone_mapping: dict[str, str | None] | None = None,
    utterance_mean_normalization: bool = True,
    global_mean_normalization: bool = True,
    frame_shift: float = 0.02,
    collapse_repeats: bool = True,
    by_gender: bool = False,
    feature_by_recording_id: bool = False,
) -> dict[str, Any]:
    """Build a phone-level reference codebook from the complete TIMIT set."""
    feature_sets = [
        (cuts_train_timit, wavlm_feat_train_timit),
        (cuts_dev_timit, wavlm_feat_dev_timit),
        (cuts_test_timit, wavlm_feat_test_timit),
    ]

    if utterance_mean_normalization:
        feature_sets = [
            (cuts, mean_normalize_wavlm_features(features))
            for cuts, features in feature_sets
        ]

    codebook_set = collect_all_speakers(feature_sets, sentence=None, by_recording_id=feature_by_recording_id)

    x_ref_all, phone_labels = build_average_phone_matrix_for_timit(
        dataset=codebook_set,
        phone_class=phone_class,
        strategy=strategy,
        by_gender=by_gender,
        phone_mapping=phone_mapping,
        frame_shift=frame_shift,
        collapse_repeats=collapse_repeats,
    )


    global_mean = None
    if global_mean_normalization:
        global_mean = x_ref_all.mean(axis=0, keepdims=True)
        x_ref_all = x_ref_all - global_mean

    u, sigma, vt = randomized_svd(
        x_ref_all,
        n_components=rank,
    )

    return {
        "U": u,
        "sigma": sigma,
        "VT": vt,
        "X_ref_all": x_ref_all,
        "phones": phone_labels,
        "global_mean": global_mean,
        "phone_class": phone_class,
        "strategy": strategy,
    }

def build_phone_codebook_from_alignment_tuple(
    datasets: Iterable[
        tuple[dict[str, np.ndarray], dict[str, Any]]
    ],
    rank: int,
    phone_class: PhoneClass = PhoneClass.DIPHONE,
    strategy: AveragingStrategy = AveragingStrategy.CENTER_PHONE,
    utterance_mean_normalization: bool = True,
    global_mean_normalization: bool = True,
    alignment_frame_shift: float = 0.01,
    feature_frame_shift: float = 0.02,
    collapse_repeats: bool = False,
    recording_ids_white_list: list = None,
) -> dict[str, Any]:
    """Build a phone codebook from arbitrary LBS feature/alignment sets.

    Parameters
    ----------
    datasets:
        Iterable of ``(features, alignments)`` pairs. Each ``features``
        dictionary maps recording IDs to feature matrices of shape
        ``(T, D)``. Each ``alignments`` dictionary maps the same recording
        IDs to tuples:

            (start_frame, end_frame, phone)

        Alignment frame indices are interpreted using
        ``alignment_frame_shift`` and converted to feature-frame indices
        using ``feature_frame_shift``.

        This can contain any number of datasets, e.g.

            [
                (features_train, alignments_train),
            ]

        or

            [
                (features_train, alignments_train),
                (features_dev, alignments_dev),
            ]

        or train/dev/test, etc.

    rank:
        Number of randomized SVD components.

    alignment_frame_shift:
        Duration in seconds represented by one alignment frame.

    feature_frame_shift:
        Duration in seconds represented by one feature frame.

    collapse_repeats:
        Whether consecutive identical phone labels in the alignment
        should be merged.
    """
    if alignment_frame_shift <= 0:
        raise ValueError(
            f"alignment_frame_shift must be positive, "
            f"got {alignment_frame_shift}"
        )

    if feature_frame_shift <= 0:
        raise ValueError(
            f"feature_frame_shift must be positive, "
            f"got {feature_frame_shift}"
        )

    phone_class = _coerce_phone_class(phone_class)
    strategy = _coerce_strategy(strategy)

    items: list[
        tuple[np.ndarray, list[tuple[int, int, str]]]
    ] = []

    for features, alignments in datasets:
        for recording_id, feat in features.items():
            if recording_ids_white_list is not None and recording_id not in recording_ids_white_list: continue
            if recording_id not in alignments:
                raise KeyError(
                    f"No alignment found for recording_id="
                    f"{recording_id!r}"
                )

            feat = np.asarray(feat)

            if feat.ndim != 2:
                raise ValueError(
                    f"Expected features for {recording_id!r} "
                    f"to have shape (T, D), got {feat.shape}"
                )

            if utterance_mean_normalization:
                feat = feat - feat.mean(
                    axis=0,
                    keepdims=True,
                )

            alignment = alignments[recording_id]

            # Convert alignment-frame indices -> seconds
            # -> feature-frame indices.
            feature_alignment = [
                (
                    int(np.floor(
                        start * alignment_frame_shift
                        / feature_frame_shift
                    )),
                    int(np.ceil(
                        end * alignment_frame_shift
                        / feature_frame_shift
                    )),
                    str(phone),
                )
                for start, end, phone in alignment
            ]

            items.append(
                (feat, feature_alignment)
            )

    x_ref_all, phone_labels = (
        build_average_phone_matrix_from_tuple_alignments(
            items=items,
            phone_class=phone_class,
            strategy=strategy,
            collapse_repeats=collapse_repeats,
        )
    )

    global_mean = None

    if global_mean_normalization:
        global_mean = x_ref_all.mean(
            axis=0,
            keepdims=True,
        )
        x_ref_all = x_ref_all - global_mean

    u, sigma, vt = randomized_svd(
        x_ref_all,
        n_components=rank,
    )

    return {
        "U": u,
        "sigma": sigma,
        "VT": vt,
        "X_ref_all": x_ref_all,
        "phones": phone_labels,
        "global_mean": global_mean,
        "phone_class": phone_class,
        "strategy": strategy,
    }



def build_phone_averages_from_pickles(
    feature_path: str | Path,
    alignments: dict[str, Any],
    phone_class: PhoneClass = PhoneClass.DIPHONE,
    strategy: AveragingStrategy = AveragingStrategy.CENTER_PHONE,
    utterance_mean_normalization: bool = True,
    collapse_repeats: bool = False,
) -> dict[str, list[Any]]:
    """
    Compute phone, diphone, or triphone averages from feature pickle chunks.

    Each pickle contains a dictionary mapping recording IDs to WavLM
    feature matrices. ``alignments`` uses the same recording IDs.

    Alignment tuples are WavLM frame indices:

        (start_frame, end_frame, phone)

    Returns
    -------
    dict[str, list[Any]]
        Mapping from recording ID to ``[X, phones]``.
    """
    feature_path = Path(feature_path)
    print(feature_path)


    pickle_files = sorted(
        feature_path.glob("*.pickle"),
        key=lambda path: int(
            path.stem.rsplit("_", 1)[1].split("-")[0]
        ),
    )

    averaged_phones: dict[str, list[Any]] = {}

    for pickle_file in pickle_files:
        print(pickle_file)

        with pickle_file.open("rb") as file:
            features = pickle.load(file)

        for recording_id, feat in features.items():
            if recording_id not in alignments:
                raise KeyError(
                    f"No alignment found for recording_id="
                    f"{recording_id!r}"
                )

            feat = np.asarray(feat)

            if feat.ndim != 2:
                raise ValueError(
                    f"Expected features for {recording_id!r} "
                    f"to have shape (T, D), got {feat.shape}"
                )

            if utterance_mean_normalization:
                feat = feat - feat.mean(
                    axis=0,
                    keepdims=True,
                )

            segments = segments_from_tuple_alignment(
                alignments[recording_id],
                collapse_repeats=collapse_repeats,
            )

            units = _build_phone_units(
                segments=segments,
                phone_class=phone_class,
                strategy=strategy,
            )

            vectors: list[FeatureVector] = []
            phones: list[PhoneLabel] = []

            for unit in units:
                vector = _average_feature_spans(
                    features=feat,
                    spans=unit.spans,
                )

                if vector is None:
                    continue

                vectors.append(vector)
                phones.append(unit.label)

            if vectors:
                X = np.stack(vectors, axis=0)
            else:
                X = np.empty(
                    (0, feat.shape[1]),
                    dtype=feat.dtype,
                )

            averaged_phones[recording_id] = [X, phones]

        del features
        gc.collect()

    return averaged_phones