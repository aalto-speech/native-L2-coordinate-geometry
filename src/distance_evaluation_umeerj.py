"""
Clean UME-ERJ evaluation utilities.

Main convention
---------------
- `score` = human rater score.
- `value` = raw system distance.
- Therefore good systems should have NEGATIVE correlations:
      corr(score, value) < 0

Main paper-style evaluation
---------------------------
1. One raw distance value per utterance/recording.
2. Compare the distance to each human rater separately.
3. Average metrics across raters.
4. Human benchmark: compare every pair of raters, then average across pairs.
5. Optional partial Spearman with controls such as n_common and silence.
6. Optional speaker-cluster bootstrap on Pearson r.

This file is intentionally self-contained except for numpy/pandas/scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from itertools import combinations
from typing import Any, Iterable

from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    rankdata,
)


###############################################################################
# Basic scalar/stat helpers
###############################################################################

def finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values) -> float:
    arr = finite_array(values)

    if arr.size == 0:
        return np.nan

    return float(np.mean(arr))


def finite_median(values) -> float:
    arr = finite_array(values)

    if arr.size == 0:
        return np.nan

    return float(np.median(arr))


def finite_std(values) -> float:
    arr = finite_array(values)

    if arr.size <= 1:
        return np.nan

    return float(np.std(arr, ddof=1))


def fisher_average_correlations(rhos) -> float:
    """
    Fisher-z average correlations.

    Not the default for paper-style macro averaging unless explicitly requested.
    """
    rhos = finite_array(rhos)

    if rhos.size == 0:
        return np.nan

    rhos = np.clip(rhos, -0.999999, 0.999999)
    z = np.arctanh(rhos)

    return float(np.tanh(np.mean(z)))


def average_correlations(values, method: str = "arithmetic") -> float:
    """
    Average correlation coefficients.

    method="arithmetic":
        Simple average. This is the default because the paper wording says
        "average across raters/pairs".

    method="fisher":
        Fisher-z average. Keep this only if you intentionally want it.
    """
    arr = finite_array(values)

    if arr.size == 0:
        return np.nan

    if method == "arithmetic":
        return float(np.mean(arr))

    if method == "fisher":
        return fisher_average_correlations(arr)

    raise ValueError("method must be 'arithmetic' or 'fisher'.")


###############################################################################
# Distance aggregation
###############################################################################

def umeerj_distance_to_scalar(
    value: Any,
    aggregation: str = "mean",
    trim_ratio: float = 0.1,
) -> float:
    """
    Convert scalar or vector-valued distance to one scalar.

    Supported aggregations:
        - "mean"
        - "median"
        - "trimmed_mean"
        - "max"
        - "min"
    """
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.nan

    if arr.size == 0:
        return np.nan

    if not np.all(np.isfinite(arr)):
        return np.nan

    if arr.ndim == 0:
        return float(arr)

    flat = arr.ravel()

    if aggregation == "mean":
        return float(np.mean(flat))

    if aggregation == "median":
        return float(np.median(flat))

    if aggregation == "max":
        return float(np.max(flat))

    if aggregation == "min":
        return float(np.min(flat))

    if aggregation == "trimmed_mean":
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must satisfy 0 <= trim_ratio < 0.5.")

        sorted_values = np.sort(flat)
        n = len(sorted_values)
        k = int(np.floor(trim_ratio * n))

        if 2 * k >= n:
            return float(np.mean(sorted_values))

        return float(np.mean(sorted_values[k:n - k]))

    raise ValueError(f"Unknown aggregation: {aggregation}")


###############################################################################
# Safe correlations
###############################################################################

def safe_corr(
    y,
    x,
    method: str = "spearman",
    min_n: int = 3,
    min_unique_y: int = 2,
    min_unique_x: int = 2,
) -> dict[str, Any]:
    """
    Safe correlation between human score y and raw distance x.

    For UME-ERJ raw distances:
        good systems should have negative correlations.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    n = int(len(y))

    out = {
        "n": n,
        "stat": np.nan,
        "p_value": np.nan,
        "reason": None,
    }

    if n < min_n:
        out["reason"] = f"n<{min_n}"
        return out

    if len(np.unique(y)) < min_unique_y:
        out["reason"] = f"unique_y<{min_unique_y}"
        return out

    if len(np.unique(x)) < min_unique_x:
        out["reason"] = f"unique_x<{min_unique_x}"
        return out

    try:
        if method == "spearman":
            stat, p_value = spearmanr(y, x)
        elif method == "kendall":
            stat, p_value = kendalltau(y, x)
        elif method == "pearson":
            stat, p_value = pearsonr(y, x)
        else:
            raise ValueError("method must be 'spearman', 'kendall', or 'pearson'.")

        out["stat"] = float(stat) if np.isfinite(stat) else np.nan
        out["p_value"] = float(p_value) if np.isfinite(p_value) else np.nan

    except Exception as exc:
        out["reason"] = str(exc)

    return out


def umeerj_corr_bundle_from_records(
    records,
    score_key: str = "score",
    value_key: str = "value",
    min_n: int = 10,
) -> dict[str, Any]:
    """
    Compute Pearson, Spearman, Kendall between human score and raw distance.

    Expected good sign:
        negative
    """
    scores = []
    values = []

    for r in records:
        try:
            score = float(r.get(score_key, np.nan))
            value = float(r.get(value_key, np.nan))
        except Exception:
            continue

        if np.isfinite(score) and np.isfinite(value):
            scores.append(score)
            values.append(value)

    pearson = safe_corr(
        scores,
        values,
        method="pearson",
        min_n=min_n,
    )

    spearman = safe_corr(
        scores,
        values,
        method="spearman",
        min_n=min_n,
    )

    kendall = safe_corr(
        scores,
        values,
        method="kendall",
        min_n=min_n,
    )

    valid = (
        np.isfinite(pearson["stat"])
        or np.isfinite(spearman["stat"])
        or np.isfinite(kendall["stat"])
    )

    reason = None

    if not valid:
        reason = (
            pearson.get("reason")
            or spearman.get("reason")
            or kendall.get("reason")
            or "all_correlations_nan"
        )

    return {
        "valid": bool(valid),
        "n": int(pearson["n"]),

        "pearson_r": pearson["stat"],
        "pearson_p_value": pearson["p_value"],

        "spearman_rho": spearman["stat"],
        "spearman_p_value": spearman["p_value"],

        "kendall_tau": kendall["stat"],
        "kendall_p_value": kendall["p_value"],

        "reason": reason,
    }


###############################################################################
# Partial Spearman
###############################################################################

def _standardize_or_none(values) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return None

    std = float(np.std(values))

    if not np.isfinite(std) or std <= 0:
        return None

    return (values - float(np.mean(values))) / std


def partial_spearman_records(
    records,
    score_key: str = "score",
    value_key: str = "value",
    controls: tuple[str, ...] = ("n_common", "silence"),
    categorical_controls: tuple[str, ...] = ("part", "fold"),
    rank_numeric_controls: bool = True,
    min_n: int = 10,
) -> dict[str, Any]:
    """
    Partial Spearman between human score and raw distance.

    Procedure:
        1. Rank score and distance.
        2. Build control matrix.
        3. Residualize both ranked variables against controls.
        4. Pearson-correlate residuals.

    Notes
    -----
    - `value` remains raw distance.
    - Good partial Spearman is expected to be negative.
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return {
            "n": 0,
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": (),
            "reason": "empty",
        }

    if score_key not in df.columns or value_key not in df.columns:
        return {
            "n": 0,
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": (),
            "reason": "missing_score_or_value",
        }

    df[score_key] = pd.to_numeric(df[score_key], errors="coerce")
    df[value_key] = pd.to_numeric(df[value_key], errors="coerce")

    if "log_n_common" in controls and "log_n_common" not in df.columns:
        if "n_common" in df.columns:
            n_common = pd.to_numeric(df["n_common"], errors="coerce")
            df["log_n_common"] = np.log1p(n_common.clip(lower=0))

    base_mask = (
        np.isfinite(df[score_key].to_numpy(dtype=float))
        & np.isfinite(df[value_key].to_numpy(dtype=float))
    )

    numeric_controls = []
    categorical_used = []

    for control in controls:
        if control not in df.columns:
            continue

        if control in categorical_controls:
            categorical_used.append(control)
            continue

        control_values = pd.to_numeric(df[control], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(control_values)

        if finite.sum() < min_n:
            continue

        if len(np.unique(control_values[finite])) < 2:
            continue

        numeric_controls.append(control)
        base_mask &= finite

    df = df.loc[base_mask].copy()

    n = int(len(df))

    if n < min_n:
        return {
            "n": n,
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": tuple(numeric_controls + categorical_used),
            "reason": f"n<{min_n}",
        }

    y = df[score_key].to_numpy(dtype=float)
    x = df[value_key].to_numpy(dtype=float)

    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {
            "n": n,
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": tuple(numeric_controls + categorical_used),
            "reason": "constant_score_or_value",
        }

    y_rank = rankdata(y)
    x_rank = rankdata(x)

    X_parts = [np.ones(n, dtype=float)]
    used_controls = ["intercept"]

    for control in numeric_controls:
        values = pd.to_numeric(df[control], errors="coerce").to_numpy(dtype=float)

        if rank_numeric_controls:
            values = rankdata(values)

        values = _standardize_or_none(values)

        if values is None:
            continue

        X_parts.append(values)
        used_controls.append(control)

    for control in categorical_used:
        dummy_df = pd.get_dummies(
            df[control].astype("string").fillna("__MISSING__"),
            prefix=control,
            drop_first=True,
            dtype=float,
        )

        for col in dummy_df.columns:
            values = dummy_df[col].to_numpy(dtype=float)

            if np.std(values) <= 0:
                continue

            X_parts.append(values)
            used_controls.append(str(col))

    X = np.column_stack(X_parts)

    try:
        beta_y, *_ = np.linalg.lstsq(X, y_rank, rcond=None)
        beta_x, *_ = np.linalg.lstsq(X, x_rank, rcond=None)

        y_resid = y_rank - X @ beta_y
        x_resid = x_rank - X @ beta_x

        if np.std(y_resid) <= 0 or np.std(x_resid) <= 0:
            raise ValueError("constant residual")

        rho, p_value = pearsonr(y_resid, x_resid)

    except Exception as exc:
        return {
            "n": n,
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": tuple(used_controls),
            "reason": str(exc),
        }

    return {
        "n": n,
        "partial_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "controls": tuple(controls),
        "used_controls": tuple(used_controls),
        "reason": None,
    }


###############################################################################
# UME-ERJ score extraction
###############################################################################

def umeerj_extract_valid_rater_scores(
    cut: Any,
    assessment_name: str = "segmental",
    raters: Iterable[str] | None = None,
    invalid_scores: tuple[int, ...] | set[int] = (-1,),
    score_values: Iterable[int] | None = None,
) -> dict[str, int]:
    """
    Extract valid per-rater scores from one Lhotse cut-like object.

    If score 0 is valid, use:
        invalid_scores=(-1,)

    If score 0 means missing, use:
        invalid_scores=(-1, 0)
    """
    if raters is not None:
        raters = set(str(r) for r in raters)

    invalid_scores = set(int(x) for x in invalid_scores)

    if score_values is not None:
        score_values = set(int(x) for x in score_values)

    try:
        supervision = cut.supervisions[0]
        assessment = supervision.custom.get("assessment", {})
        score_dict = assessment.get(assessment_name, None)
    except Exception:
        return {}

    if not isinstance(score_dict, dict):
        return {}

    out = {}

    for rater, raw_score in score_dict.items():
        rater = str(rater)

        if raters is not None and rater not in raters:
            continue

        if raw_score is None:
            continue

        try:
            score = int(raw_score)
        except Exception:
            continue

        if score in invalid_scores:
            continue

        if score_values is not None and score not in score_values:
            continue

        out[rater] = score

    return out


def umeerj_collect_scores_by_rater_recording(
    cuts: Iterable[Any],
    indices: set[int] | None = None,
    assessment_name: str = "segmental",
    raters: Iterable[str] | None = None,
    invalid_scores: tuple[int, ...] | set[int] = (-1,),
    score_values: Iterable[int] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Return:
        rater -> recording_id -> score
    """
    scores_by_rater: dict[str, dict[str, int]] = {}

    for idx, cut in enumerate(cuts):
        if indices is not None and idx not in indices:
            continue

        recording_id = str(cut.recording_id)

        scores = umeerj_extract_valid_rater_scores(
            cut=cut,
            assessment_name=assessment_name,
            raters=raters,
            invalid_scores=invalid_scores,
            score_values=score_values,
        )

        for rater, score in scores.items():
            scores_by_rater.setdefault(str(rater), {})
            scores_by_rater[str(rater)][recording_id] = int(score)

    return scores_by_rater


def umeerj_collect_rating_records(
    cuts: Iterable[Any],
    indices: set[int] | None = None,
    assessment_name: str = "segmental",
    raters: Iterable[str] | None = None,
    invalid_scores: tuple[int, ...] | set[int] = (-1,),
    score_values: Iterable[int] | None = None,
    speaker_by_recording: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return long rating table records:

        recording_id, speaker_id, rater, score

    Use this if you want a separate rating table.
    The main config evaluator can also work from scores_by_rater.
    """
    records = []

    for idx, cut in enumerate(cuts):
        if indices is not None and idx not in indices:
            continue

        recording_id = str(cut.recording_id)

        speaker_id = None

        if speaker_by_recording is not None:
            speaker_id = speaker_by_recording.get(recording_id)

        if speaker_id is None:
            try:
                supervision = cut.supervisions[0]
                speaker_id = getattr(supervision, "speaker", None)

                if speaker_id is None:
                    speaker_id = supervision.custom.get("speaker_id", None)
            except Exception:
                pass

        if speaker_id is None:
            try:
                speaker_id = getattr(cut, "speaker", None)
            except Exception:
                pass

        if speaker_id is None:
            speaker_id = recording_id

        scores = umeerj_extract_valid_rater_scores(
            cut=cut,
            assessment_name=assessment_name,
            raters=raters,
            invalid_scores=invalid_scores,
            score_values=score_values,
        )

        for rater, score in scores.items():
            records.append(
                {
                    "recording_id": recording_id,
                    "speaker_id": str(speaker_id),
                    "rater": str(rater),
                    "score": float(score),
                }
            )

    return records


def collect_umeerj_scores_for_datasets(
    cuts: dict[str, object],
    datasets: tuple[str, ...] | list[str],
    indices: dict[str, set[int]] | None = None,
    assessment_name: str = "segmental",
    raters: Iterable[str] | None = None,
    invalid_scores: tuple[int, ...] | set[int] = (-1,),
    score_values: Iterable[int] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Convenience dataset wrapper.

    Return:
        dataset -> rater -> recording_id -> score
    """
    scores_by_dataset = {}

    for dataset in datasets:
        dataset_indices = None if indices is None else indices.get(dataset)

        scores_by_dataset[dataset] = umeerj_collect_scores_by_rater_recording(
            cuts=cuts[dataset],
            indices=dataset_indices,
            assessment_name=assessment_name,
            raters=raters,
            invalid_scores=invalid_scores,
            score_values=score_values,
        )

        n_raters = len(scores_by_dataset[dataset])
        n_ratings = sum(len(v) for v in scores_by_dataset[dataset].values())

        print(
            f"{dataset}: collected {n_ratings} ratings "
            f"from {n_raters} raters"
        )

    return scores_by_dataset


###############################################################################
# Distance record collection
###############################################################################

def umeerj_resolve_recording_distances(
    distances_by_model,
    encoder,
    layer,
    dataset,
    strategy,
    phone_class,
    alignment,
    rank,
):
    """
    Resolve nested distance structure:

        distances_by_model[encoder][layer][dataset]
            [strategy][phone_class][alignment][rank]
            [recording_id][metric]

    Returns:
        rank_key, recording_distances
    """
    distance_grid = distances_by_model[encoder][layer][dataset]
    align_data = distance_grid[strategy][phone_class][alignment]

    candidates = [rank, str(rank)]

    try:
        candidates.append(int(rank))
    except Exception:
        pass

    for candidate in candidates:
        if candidate in align_data:
            return candidate, align_data[candidate]

    raise KeyError(
        f"Rank {rank} not found. Available ranks: {list(align_data.keys())}"
    )


def _extract_silence(
    recording_id: str,
    silence_by_recording: dict[str, Any] | None,
    silence_key: str = "silence_ratio",
) -> float:
    if silence_by_recording is None:
        return np.nan

    if recording_id not in silence_by_recording:
        return np.nan

    raw = silence_by_recording[recording_id]

    if isinstance(raw, dict):
        raw = raw.get(silence_key, np.nan)

    try:
        value = float(raw)
    except Exception:
        value = np.nan

    return value if np.isfinite(value) else np.nan


def umeerj_collect_distance_rater_records_clean(
    recording_distances: dict[str, dict[str, Any]],
    scores_by_rater: dict[str, dict[str, int]],
    metric: str,
    phone_aggregation: str = "mean",
    trim_ratio: float = 0.1,
    speaker_by_recording: dict[str, str] | None = None,
    silence_by_recording: dict[str, Any] | None = None,
    silence_key: str = "silence_ratio",
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Build canonical UME-ERJ rater-level records.

    Output format:
        recording_id
        speaker_id
        rater
        score        # human score
        value        # raw distance, no sign flip
        n_common
        silence
        metric
        plus config keys

    Expected good sign:
        corr(score, value) < 0
    """
    records = []

    if config is None:
        config = {}

    for rater, scores_for_rater in scores_by_rater.items():
        for recording_id, score in scores_for_rater.items():
            recording_id = str(recording_id)

            if recording_id not in recording_distances:
                continue

            metric_dict = recording_distances[recording_id]

            if metric not in metric_dict:
                continue

            distance_value = umeerj_distance_to_scalar(
                value=metric_dict[metric],
                aggregation=phone_aggregation,
                trim_ratio=trim_ratio,
            )

            if not np.isfinite(distance_value):
                continue

            n_common = np.nan

            if "common_phones" in metric_dict:
                try:
                    n_common = len(metric_dict["common_phones"])
                except Exception:
                    n_common = np.nan

            silence = _extract_silence(
                recording_id=recording_id,
                silence_by_recording=silence_by_recording,
                silence_key=silence_key,
            )

            if speaker_by_recording is not None:
                speaker_id = speaker_by_recording.get(recording_id, recording_id)
            else:
                # Fallback. For real speaker-cluster bootstrap, pass true speakers.
                speaker_id = recording_id

            item = {
                "recording_id": recording_id,
                "speaker_id": str(speaker_id),
                "rater": str(rater),
                "score": float(score),
                "value": float(distance_value),
                "raw_distance": float(distance_value),
                "n_common": n_common,
                "silence": silence,
                "metric": metric,
            }

            item.update(config)
            records.append(item)

    return records


def umeerj_get_available_distance_metrics(
    recording_distances: dict[str, dict[str, Any]],
    exclude_keys: set[str] | None = None,
) -> list[str]:
    """
    Get all available distance metric names from recording_distances.
    """
    if exclude_keys is None:
        exclude_keys = {"common_phones"}

    metrics = set()

    for metric_dict in recording_distances.values():
        if not isinstance(metric_dict, dict):
            continue

        for key in metric_dict:
            if key not in exclude_keys:
                metrics.add(str(key))

    return sorted(metrics)


###############################################################################
# Main model-distance-vs-rater evaluation
###############################################################################

def umeerj_evaluate_records_by_rater_clean(
    records,
    rater_key: str = "rater",
    score_key: str = "score",
    value_key: str = "value",
    controls: tuple[str, ...] | None = ("n_common", "silence"),
    min_n: int = 10,
    average_method: str = "arithmetic",
) -> dict[str, Any]:
    """
    Paper-style distance evaluation:

        for each rater:
            corr(human_score, raw_distance)

        then average across raters.

    Expected good sign:
        negative
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return {
            "macro": {
                "n_raters": 0,
                "macro_pearson_r": np.nan,
                "macro_spearman_rho": np.nan,
                "macro_kendall_tau": np.nan,
                "macro_partial_spearman_rho": np.nan,
                "expected_good_sign": "negative",
                "average_method": average_method,
            },
            "per_rater": {},
        }

    if rater_key not in df.columns:
        raise KeyError(f"Missing rater_key='{rater_key}'.")

    per_rater = {}

    for rater, g in df.groupby(rater_key, sort=True):
        rater_records = g.to_dict("records")

        corr = umeerj_corr_bundle_from_records(
            rater_records,
            score_key=score_key,
            value_key=value_key,
            min_n=min_n,
        )

        if controls is not None and corr["n"] >= min_n:
            partial = partial_spearman_records(
                rater_records,
                score_key=score_key,
                value_key=value_key,
                controls=controls,
                min_n=min_n,
            )
        else:
            partial = {
                "n": corr["n"],
                "partial_spearman_rho": np.nan,
                "p_value": np.nan,
                "controls": tuple(controls) if controls is not None else (),
                "used_controls": (),
                "reason": "controls_disabled_or_n_too_small",
            }

        per_rater[str(rater)] = {
            **corr,

            "partial_spearman_rho": partial["partial_spearman_rho"],
            "partial_p_value": partial["p_value"],
            "partial_controls": partial["controls"],
            "used_controls": partial.get("used_controls", ()),
            "partial_reason": partial.get("reason"),
        }

    pearsons = [x["pearson_r"] for x in per_rater.values()]
    spearmans = [x["spearman_rho"] for x in per_rater.values()]
    kendalls = [x["kendall_tau"] for x in per_rater.values()]
    partials = [x["partial_spearman_rho"] for x in per_rater.values()]

    macro = {
        "n_raters": int(len(per_rater)),

        "macro_pearson_r": average_correlations(
            pearsons,
            method=average_method,
        ),
        "macro_spearman_rho": average_correlations(
            spearmans,
            method=average_method,
        ),
        "macro_kendall_tau": average_correlations(
            kendalls,
            method=average_method,
        ),
        "macro_partial_spearman_rho": average_correlations(
            partials,
            method=average_method,
        ),

        "median_pearson_r": finite_median(pearsons),
        "median_spearman_rho": finite_median(spearmans),
        "median_kendall_tau": finite_median(kendalls),
        "median_partial_spearman_rho": finite_median(partials),

        "std_pearson_r": finite_std(pearsons),
        "std_spearman_rho": finite_std(spearmans),
        "std_kendall_tau": finite_std(kendalls),
        "std_partial_spearman_rho": finite_std(partials),

        "n_valid_pearson_r": int(np.isfinite(np.asarray(pearsons, dtype=float)).sum()),
        "n_valid_spearman_rho": int(np.isfinite(np.asarray(spearmans, dtype=float)).sum()),
        "n_valid_kendall_tau": int(np.isfinite(np.asarray(kendalls, dtype=float)).sum()),
        "n_valid_partial_spearman_rho": int(np.isfinite(np.asarray(partials, dtype=float)).sum()),

        "average_method": average_method,
        "orientation": "raw_distance",
        "expected_good_sign": "negative",
    }

    return {
        "macro": macro,
        "per_rater": per_rater,
    }


###############################################################################
# Human-human benchmark
###############################################################################

def _pearson_from_sufficient_stats(stats: np.ndarray, min_n: int = 10) -> np.ndarray:
    """
    Compute Pearson r from sufficient statistics.

    stats[..., 0] = n
    stats[..., 1] = sum_x
    stats[..., 2] = sum_y
    stats[..., 3] = sum_xx
    stats[..., 4] = sum_yy
    stats[..., 5] = sum_xy
    """
    stats = np.asarray(stats, dtype=float)

    n = stats[..., 0]
    sx = stats[..., 1]
    sy = stats[..., 2]
    sxx = stats[..., 3]
    syy = stats[..., 4]
    sxy = stats[..., 5]

    numerator = n * sxy - sx * sy

    denom_x = n * sxx - sx * sx
    denom_y = n * syy - sy * sy

    denominator = np.sqrt(denom_x * denom_y)

    r = np.full_like(n, np.nan, dtype=float)

    valid = (
        (n >= min_n)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    r[valid] = numerator[valid] / denominator[valid]

    return r


def _sufficient_stats_from_xy(x, y) -> np.ndarray:
    """
    Return sufficient stats for Pearson correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6, dtype=float)

    return np.asarray(
        [
            float(x.size),
            float(np.sum(x)),
            float(np.sum(y)),
            float(np.sum(x * x)),
            float(np.sum(y * y)),
            float(np.sum(x * y)),
        ],
        dtype=float,
    )


def _macro_from_r_values(
    r_values: np.ndarray,
    average_method: str = "arithmetic",
) -> np.ndarray:
    """
    Macro-average correlations along the last axis.

    Supports both:
        shape = (n_items,)
        shape = (n_boot, n_items)
    """
    r = np.asarray(r_values, dtype=float)

    if r.ndim == 1:
        arr = r[np.isfinite(r)]

        if arr.size == 0:
            return np.asarray(np.nan)

        if average_method == "arithmetic":
            return np.asarray(float(np.mean(arr)))

        if average_method == "fisher":
            arr = np.clip(arr, -0.999999, 0.999999)
            return np.asarray(float(np.tanh(np.mean(np.arctanh(arr)))))

        raise ValueError("average_method must be 'arithmetic' or 'fisher'.")

    if r.ndim == 2:
        mask = np.isfinite(r)

        if average_method == "arithmetic":
            sums = np.where(mask, r, 0.0).sum(axis=1)
            counts = mask.sum(axis=1)

            out = np.full(r.shape[0], np.nan, dtype=float)
            valid = counts > 0
            out[valid] = sums[valid] / counts[valid]
            return out

        if average_method == "fisher":
            clipped = np.clip(r, -0.999999, 0.999999)
            z = np.where(mask, np.arctanh(clipped), 0.0)

            sums = z.sum(axis=1)
            counts = mask.sum(axis=1)

            out = np.full(r.shape[0], np.nan, dtype=float)
            valid = counts > 0
            out[valid] = np.tanh(sums[valid] / counts[valid])
            return out

        raise ValueError("average_method must be 'arithmetic' or 'fisher'.")

    raise ValueError("r_values must be 1D or 2D.")



def umeerj_fast_speaker_cluster_bootstrap_r(
    records,
    cluster_key: str = "speaker_id",
    item_key: str = "recording_id",
    rater_key: str = "rater",
    score_key: str = "score",
    value_key: str = "value",
    n_boot: int = 10000,
    seed: int = 0,
    min_n: int = 10,
    average_method: str = "arithmetic",
    include_human: bool = True,
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    Fast speaker-cluster bootstrap on Pearson r.

    This is much faster than the pandas-loop bootstrap because it precomputes
    Pearson sufficient statistics per cluster and then only resamples/sums stats.

    Raw-distance convention:
        model r = corr(score, raw_distance)

    Expected:
        model r < 0
        human-human r > 0
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return {
            "model_macro_pearson_r": _bootstrap_ci([], np.nan),
            "human_macro_pearson_r": _bootstrap_ci([], np.nan),
            "distance_strength_macro_pearson_r": _bootstrap_ci([], np.nan),
            "diagnostics": {
                "n_records": 0,
                "n_clusters": 0,
                "fast": True,
            },
        }

    required = [cluster_key, item_key, rater_key, score_key, value_key]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns for fast bootstrap: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df[cluster_key] = df[cluster_key].astype(str)
    df[item_key] = df[item_key].astype(str)
    df[rater_key] = df[rater_key].astype(str)

    df[score_key] = pd.to_numeric(df[score_key], errors="coerce")
    df[value_key] = pd.to_numeric(df[value_key], errors="coerce")

    finite = (
        np.isfinite(df[score_key].to_numpy(dtype=float))
        & np.isfinite(df[value_key].to_numpy(dtype=float))
    )

    df = df.loc[finite].copy()

    cluster_ids = sorted(df[cluster_key].unique(), key=str)
    raters = sorted(df[rater_key].unique(), key=str)

    n_clusters = len(cluster_ids)
    n_raters = len(raters)

    if n_clusters == 0 or n_raters == 0:
        return {
            "model_macro_pearson_r": _bootstrap_ci([], np.nan),
            "human_macro_pearson_r": _bootstrap_ci([], np.nan),
            "distance_strength_macro_pearson_r": _bootstrap_ci([], np.nan),
            "diagnostics": {
                "n_records": int(len(df)),
                "n_clusters": int(n_clusters),
                "n_raters": int(n_raters),
                "fast": True,
            },
        }

    cluster_to_idx = {
        cluster: i
        for i, cluster in enumerate(cluster_ids)
    }

    rater_to_idx = {
        rater: i
        for i, rater in enumerate(raters)
    }

    # ------------------------------------------------------------------
    # Model-vs-rater sufficient stats
    # ------------------------------------------------------------------
    model_stats = np.zeros(
        (n_clusters, n_raters, 6),
        dtype=float,
    )

    for (cluster, rater), g in df.groupby([cluster_key, rater_key], sort=False):
        ci = cluster_to_idx[str(cluster)]
        ri = rater_to_idx[str(rater)]

        x = g[score_key].to_numpy(dtype=float)
        y = g[value_key].to_numpy(dtype=float)

        model_stats[ci, ri, :] = _sufficient_stats_from_xy(x, y)

    # ------------------------------------------------------------------
    # Human-human sufficient stats
    # ------------------------------------------------------------------
    if include_human:
        rater_pairs = list(combinations(raters, 2))

        pair_to_idx = {
            pair: i
            for i, pair in enumerate(rater_pairs)
        }

        n_pairs = len(rater_pairs)

        human_stats = np.zeros(
            (n_clusters, n_pairs, 6),
            dtype=float,
        )


        # One score per cluster/item/rater.
        # Important: cluster_key and item_key may be the same, e.g.
        # cluster_key="recording_id", item_key="recording_id".
        rating_cols = []
        for col in [cluster_key, item_key, rater_key, score_key]:
            if col not in rating_cols:
                rating_cols.append(col)

        group_cols = []
        for col in [cluster_key, item_key, rater_key]:
            if col not in group_cols:
                group_cols.append(col)

        rating_df = (
            df[rating_cols]
                .groupby(group_cols, as_index=False)
                .agg({score_key: "mean"})
        )

        for cluster, cluster_df in rating_df.groupby(cluster_key, sort=False):
            ci = cluster_to_idx[str(cluster)]

            for _, item_df in cluster_df.groupby(item_key, sort=False):
                score_by_rater = {
                    str(row[rater_key]): float(row[score_key])
                    for _, row in item_df.iterrows()
                    if np.isfinite(float(row[score_key]))
                }

                present_raters = sorted(score_by_rater.keys(), key=str)

                for rater_a, rater_b in combinations(present_raters, 2):
                    pair = (rater_a, rater_b)

                    if pair not in pair_to_idx:
                        continue

                    pi = pair_to_idx[pair]

                    x = score_by_rater[rater_a]
                    y = score_by_rater[rater_b]

                    human_stats[ci, pi, :] += _sufficient_stats_from_xy(
                        [x],
                        [y],
                    )
    else:
        rater_pairs = []
        human_stats = None

    # ------------------------------------------------------------------
    # Point estimates
    # ------------------------------------------------------------------
    point_model_stats = model_stats.sum(axis=0)
    point_model_r_by_rater = _pearson_from_sufficient_stats(
        point_model_stats,
        min_n=min_n,
    )

    point_model_r = float(
        _macro_from_r_values(
            point_model_r_by_rater,
            average_method=average_method,
        )
    )

    point_distance_strength = (
        -point_model_r
        if np.isfinite(point_model_r)
        else np.nan
    )

    if include_human:
        point_human_stats = human_stats.sum(axis=0)
        point_human_r_by_pair = _pearson_from_sufficient_stats(
            point_human_stats,
            min_n=min_n,
        )

        point_human_r = float(
            _macro_from_r_values(
                point_human_r_by_pair,
                average_method=average_method,
            )
        )
    else:
        point_human_r = np.nan

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    model_boot = []
    strength_boot = []
    human_boot = []

    probs = np.full(
        n_clusters,
        1.0 / n_clusters,
        dtype=float,
    )

    n_done = 0

    while n_done < n_boot:
        current_batch = min(
            batch_size,
            n_boot - n_done,
        )

        # Multinomial counts are equivalent to sampling clusters with replacement.
        counts = rng.multinomial(
            n=n_clusters,
            pvals=probs,
            size=current_batch,
        ).astype(float)

        # Shape:
        #   counts:      batch x clusters
        #   model_stats: clusters x raters x 6
        #   boot_model_stats: batch x raters x 6
        boot_model_stats = np.einsum(
            "bc,crs->brs",
            counts,
            model_stats,
            optimize=True,
        )

        boot_model_r_by_rater = _pearson_from_sufficient_stats(
            boot_model_stats,
            min_n=min_n,
        )

        boot_model_macro = _macro_from_r_values(
            boot_model_r_by_rater,
            average_method=average_method,
        )

        for val in boot_model_macro:
            if np.isfinite(val):
                model_boot.append(float(val))
                strength_boot.append(float(-val))

        if include_human:
            boot_human_stats = np.einsum(
                "bc,cps->bps",
                counts,
                human_stats,
                optimize=True,
            )

            boot_human_r_by_pair = _pearson_from_sufficient_stats(
                boot_human_stats,
                min_n=min_n,
            )

            boot_human_macro = _macro_from_r_values(
                boot_human_r_by_pair,
                average_method=average_method,
            )

            for val in boot_human_macro:
                if np.isfinite(val):
                    human_boot.append(float(val))

        n_done += current_batch

    model_arr = finite_array(model_boot)
    strength_arr = finite_array(strength_boot)

    if model_arr.size > 0:
        p_negative_vs_zero = float(
            (1.0 + np.sum(model_arr >= 0.0)) / (1.0 + model_arr.size)
        )
    else:
        p_negative_vs_zero = np.nan

    if strength_arr.size > 0:
        p_strength_positive_vs_zero = float(
            (1.0 + np.sum(strength_arr <= 0.0)) / (1.0 + strength_arr.size)
        )
    else:
        p_strength_positive_vs_zero = np.nan

    return {
        "model_macro_pearson_r": {
            **_bootstrap_ci(model_boot, point_model_r),
            "expected_good_sign": "negative",
            "p_negative_vs_zero": p_negative_vs_zero,
        },
        "human_macro_pearson_r": {
            **_bootstrap_ci(human_boot, point_human_r),
            "expected_good_sign": "positive",
        },
        "distance_strength_macro_pearson_r": {
            **_bootstrap_ci(strength_boot, point_distance_strength),
            "definition": "-model_macro_pearson_r",
            "expected_good_sign": "positive",
            "p_positive_vs_zero": p_strength_positive_vs_zero,
        },
        "diagnostics": {
            "n_records": int(len(df)),
            "n_clusters": int(n_clusters),
            "n_raters": int(n_raters),
            "n_pairs": int(len(rater_pairs)),
            "cluster_key": cluster_key,
            "item_key": item_key,
            "n_boot": int(n_boot),
            "min_n": int(min_n),
            "average_method": average_method,
            "include_human": bool(include_human),
            "fast": True,
        },
    }

def umeerj_human_benchmark_from_records_clean(
    records,
    item_keys: tuple[str, ...] = ("recording_id",),
    rater_key: str = "rater",
    score_key: str = "score",
    min_n: int = 10,
    average_method: str = "arithmetic",
) -> dict[str, Any]:
    """
    Human-human benchmark:

        for every pair of raters:
            corr(score_rater_a, score_rater_b)

        then average over rater pairs.

    Expected good sign:
        positive
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return {
            "macro": {
                "n_pairs": 0,
                "macro_pearson_r": np.nan,
                "macro_spearman_rho": np.nan,
                "macro_kendall_tau": np.nan,
                "expected_good_sign": "positive",
                "average_method": average_method,
            },
            "per_pair": {},
        }

    item_keys = tuple(item_keys)

    needed = list(item_keys) + [rater_key, score_key]

    missing = [col for col in needed if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns for human benchmark: {missing}")

    df = df[needed].copy()
    df[score_key] = pd.to_numeric(df[score_key], errors="coerce")
    df = df[np.isfinite(df[score_key].to_numpy(dtype=float))].copy()

    # Prevent accidental duplicates from breaking pairwise merge.
    df = (
        df.groupby(list(item_keys) + [rater_key], as_index=False)
        .agg({score_key: "mean"})
    )

    raters = sorted(df[rater_key].astype(str).unique())
    per_pair = {}

    for rater_a, rater_b in combinations(raters, 2):
        da = df[df[rater_key].astype(str) == rater_a]
        db = df[df[rater_key].astype(str) == rater_b]

        pair_df = da.merge(
            db,
            on=list(item_keys),
            how="inner",
            suffixes=("_a", "_b"),
        )

        pair_records = [
            {
                "score": row[f"{score_key}_a"],
                "value": row[f"{score_key}_b"],
            }
            for _, row in pair_df.iterrows()
        ]

        corr = umeerj_corr_bundle_from_records(
            pair_records,
            score_key="score",
            value_key="value",
            min_n=min_n,
        )

        key = f"{rater_a}__{rater_b}"

        per_pair[key] = {
            **corr,
            "rater_a": rater_a,
            "rater_b": rater_b,
        }

    pearsons = [x["pearson_r"] for x in per_pair.values()]
    spearmans = [x["spearman_rho"] for x in per_pair.values()]
    kendalls = [x["kendall_tau"] for x in per_pair.values()]

    macro = {
        "n_pairs": int(len(per_pair)),

        "macro_pearson_r": average_correlations(
            pearsons,
            method=average_method,
        ),
        "macro_spearman_rho": average_correlations(
            spearmans,
            method=average_method,
        ),
        "macro_kendall_tau": average_correlations(
            kendalls,
            method=average_method,
        ),

        "median_pearson_r": finite_median(pearsons),
        "median_spearman_rho": finite_median(spearmans),
        "median_kendall_tau": finite_median(kendalls),

        "std_pearson_r": finite_std(pearsons),
        "std_spearman_rho": finite_std(spearmans),
        "std_kendall_tau": finite_std(kendalls),

        "n_valid_pearson_r": int(np.isfinite(np.asarray(pearsons, dtype=float)).sum()),
        "n_valid_spearman_rho": int(np.isfinite(np.asarray(spearmans, dtype=float)).sum()),
        "n_valid_kendall_tau": int(np.isfinite(np.asarray(kendalls, dtype=float)).sum()),

        "average_method": average_method,
        "expected_good_sign": "positive",
    }

    return {
        "macro": macro,
        "per_pair": per_pair,
    }


###############################################################################
# Auxiliary diagnostics
###############################################################################

def build_record_diagnostics(
    records,
    recording_id_key: str = "recording_id",
    speaker_id_key: str = "speaker_id",
    rater_key: str = "rater",
    numeric_cols: tuple[str, ...] = ("score", "value", "n_common", "silence"),
) -> dict[str, Any]:
    """
    Diagnostics for canonical UME-ERJ records.
    """
    df = pd.DataFrame(records).copy()

    diagnostics: dict[str, Any] = {
        "n_records": int(len(df)),
    }

    if recording_id_key in df.columns:
        diagnostics["n_unique_recordings"] = int(df[recording_id_key].nunique())
    else:
        diagnostics["n_unique_recordings"] = 0

    if speaker_id_key in df.columns:
        diagnostics["n_unique_speakers"] = int(df[speaker_id_key].nunique())
    else:
        diagnostics["n_unique_speakers"] = 0

    if rater_key in df.columns:
        diagnostics["n_raters"] = int(df[rater_key].nunique())
    else:
        diagnostics["n_raters"] = 0

    for col in numeric_cols:
        if col not in df.columns:
            diagnostics[f"finite_{col}"] = 0
            diagnostics[f"unique_{col}"] = 0
            continue

        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)

        diagnostics[f"finite_{col}"] = int(finite.sum())
        diagnostics[f"unique_{col}"] = (
            int(len(np.unique(values[finite])))
            if np.any(finite)
            else 0
        )

        if np.any(finite):
            diagnostics[f"min_{col}"] = float(np.min(values[finite]))
            diagnostics[f"mean_{col}"] = float(np.mean(values[finite]))
            diagnostics[f"max_{col}"] = float(np.max(values[finite]))

    if rater_key in df.columns:
        per_rater = []

        for rater, g in df.groupby(rater_key, sort=True):
            item = {
                "rater": str(rater),
                "n": int(len(g)),
            }

            if recording_id_key in g.columns:
                item["n_recordings"] = int(g[recording_id_key].nunique())

            for col in ["score", "value"]:
                if col in g.columns:
                    arr = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                    arr = arr[np.isfinite(arr)]
                    item[f"unique_{col}"] = int(len(np.unique(arr))) if arr.size else 0

            per_rater.append(item)

        diagnostics["per_rater"] = per_rater

    return diagnostics


def umeerj_auxiliary_associations_by_rater_clean(
    records,
    aux_names: tuple[str, ...] = ("n_common", "silence"),
    rater_key: str = "rater",
    score_key: str = "score",
    min_n: int = 10,
    average_method: str = "arithmetic",
) -> dict[str, Any]:
    """
    Correlate auxiliary variables with human score, per rater.

    Useful checks:
        - Does n_common correlate with proficiency?
        - Does silence correlate with proficiency?
    """
    output = {}

    for aux_name in aux_names:
        aux_records = []

        for r in records:
            aux_value = r.get(aux_name, np.nan)

            try:
                aux_value = float(aux_value)
            except Exception:
                continue

            if not np.isfinite(aux_value):
                continue

            rr = dict(r)
            rr["value"] = aux_value
            aux_records.append(rr)

        output[aux_name] = umeerj_evaluate_records_by_rater_clean(
            aux_records,
            rater_key=rater_key,
            score_key=score_key,
            value_key="value",
            controls=None,
            min_n=min_n,
            average_method=average_method,
        )

    return output


###############################################################################
# Speaker-cluster bootstrap on Pearson r
###############################################################################

def _bootstrap_ci(values, point) -> dict[str, Any]:
    arr = finite_array(values)

    if arr.size == 0:
        return {
            "point": float(point) if np.isfinite(point) else np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot_valid": 0,
        }

    lo, hi = np.percentile(arr, [2.5, 97.5])

    return {
        "point": float(point) if np.isfinite(point) else np.nan,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot_valid": int(arr.size),
    }


def umeerj_speaker_cluster_bootstrap_r_clean(
    records,
    cluster_key: str = "speaker_id",
    rater_key: str = "rater",
    score_key: str = "score",
    value_key: str = "value",
    n_boot: int = 10000,
    seed: int = 0,
    min_n: int = 10,
    average_method: str = "arithmetic",
) -> dict[str, Any]:
    """
    Paired speaker-cluster bootstrap on Pearson r.

    Uses raw distances:
        model r = corr(score, raw_distance)

    Expected:
        model r < 0
        human-human r > 0
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return {
            "model_macro_pearson_r": _bootstrap_ci([], np.nan),
            "human_macro_pearson_r": _bootstrap_ci([], np.nan),
            "distance_strength_macro_pearson_r": _bootstrap_ci([], np.nan),
            "diagnostics": {
                "n_records": 0,
                "n_clusters": 0,
            },
        }

    if cluster_key not in df.columns:
        raise KeyError(
            f"cluster_key='{cluster_key}' missing. "
            f"Available columns: {list(df.columns)}"
        )

    df[cluster_key] = df[cluster_key].astype(str)

    cluster_ids = sorted(df[cluster_key].unique())

    if len(cluster_ids) == 0:
        return {
            "model_macro_pearson_r": _bootstrap_ci([], np.nan),
            "human_macro_pearson_r": _bootstrap_ci([], np.nan),
            "distance_strength_macro_pearson_r": _bootstrap_ci([], np.nan),
            "diagnostics": {
                "n_records": int(len(df)),
                "n_clusters": 0,
            },
        }

    def evaluate_model_r(sample_records):
        return umeerj_evaluate_records_by_rater_clean(
            sample_records,
            rater_key=rater_key,
            score_key=score_key,
            value_key=value_key,
            controls=None,
            min_n=min_n,
            average_method=average_method,
        )["macro"]["macro_pearson_r"]

    def evaluate_human_r(sample_records, item_keys=("recording_id",)):
        return umeerj_human_benchmark_from_records_clean(
            sample_records,
            item_keys=item_keys,
            rater_key=rater_key,
            score_key=score_key,
            min_n=min_n,
            average_method=average_method,
        )["macro"]["macro_pearson_r"]

    point_records = df.to_dict("records")

    point_model_r = evaluate_model_r(point_records)
    point_human_r = evaluate_human_r(point_records)

    # Since raw distance gives negative model r, distance strength is -r.
    point_distance_strength = (
        -point_model_r
        if np.isfinite(point_model_r)
        else np.nan
    )

    cluster_to_df = {
        cluster_id: g.copy()
        for cluster_id, g in df.groupby(cluster_key, sort=False)
    }

    rng = np.random.default_rng(seed)

    model_boot = []
    human_boot = []
    strength_boot = []

    for _ in range(n_boot):
        sampled_clusters = rng.choice(
            cluster_ids,
            size=len(cluster_ids),
            replace=True,
        )

        pieces = []

        for draw_idx, cluster_id in enumerate(sampled_clusters):
            g = cluster_to_df[str(cluster_id)].copy()

            # Needed so duplicated bootstrap speakers do not cause many-to-many
            # rater-pair collisions.
            g["_boot_draw"] = draw_idx

            pieces.append(g)

        sample_df = pd.concat(pieces, ignore_index=True)
        sample_records = sample_df.to_dict("records")

        model_r = evaluate_model_r(sample_records)

        human_r = evaluate_human_r(
            sample_records,
            item_keys=("_boot_draw", "recording_id"),
        )

        if np.isfinite(model_r):
            model_boot.append(float(model_r))
            strength_boot.append(float(-model_r))

        if np.isfinite(human_r):
            human_boot.append(float(human_r))

    model_arr = finite_array(model_boot)
    strength_arr = finite_array(strength_boot)

    if model_arr.size > 0:
        # One-sided p-value for expected negative raw-distance correlation.
        p_negative_vs_zero = float(
            (1.0 + np.sum(model_arr >= 0.0)) / (1.0 + model_arr.size)
        )
    else:
        p_negative_vs_zero = np.nan

    if strength_arr.size > 0:
        # Equivalent one-sided p-value for distance strength > 0.
        p_strength_positive_vs_zero = float(
            (1.0 + np.sum(strength_arr <= 0.0)) / (1.0 + strength_arr.size)
        )
    else:
        p_strength_positive_vs_zero = np.nan

    return {
        "model_macro_pearson_r": {
            **_bootstrap_ci(model_boot, point_model_r),
            "expected_good_sign": "negative",
            "p_negative_vs_zero": p_negative_vs_zero,
        },
        "human_macro_pearson_r": {
            **_bootstrap_ci(human_boot, point_human_r),
            "expected_good_sign": "positive",
        },
        "distance_strength_macro_pearson_r": {
            **_bootstrap_ci(strength_boot, point_distance_strength),
            "definition": "-model_macro_pearson_r",
            "expected_good_sign": "positive",
            "p_positive_vs_zero": p_strength_positive_vs_zero,
        },
        "diagnostics": {
            "n_records": int(len(df)),
            "n_clusters": int(len(cluster_ids)),
            "cluster_key": cluster_key,
            "n_boot": int(n_boot),
            "min_n": int(min_n),
            "average_method": average_method,
        },
    }


###############################################################################
# One-config evaluator
###############################################################################

def umeerj_evaluate_one_config_clean(
    distances_by_model: dict[str, dict[int, dict[str, dict]]],
    scores_by_dataset: dict[str, dict[str, dict[str, int]]],
    encoder: str,
    layer: int,
    dataset: str,
    strategy: str,
    phone_class: str,
    alignment: str,
    rank: Any,
    metric: str,
    phone_aggregation: str = "mean",
    trim_ratio: float = 0.1,
    speaker_by_recording: dict[str, str] | None = None,
    silence_by_recording: dict[str, Any] | None = None,
    silence_key: str = "silence_ratio",
    controls: tuple[str, ...] | None = ("n_common", "silence"),
    min_n: int = 10,
    average_method: str = "arithmetic",
    bootstrap: bool = False,
    n_boot: int = 10000,
    seed: int = 0,
    return_records: bool = True,
) -> dict[str, Any]:
    """
    Clean UME-ERJ evaluation for one distance configuration.

    Preserved logic:
        - raw distance is used
        - expected good sign is negative
        - per-rater correlations are computed
        - macro averages over raters
        - human benchmark averages over rater pairs
        - partial Spearman is optional/diagnostic
        - optional speaker-cluster bootstrap on Pearson r
    """
    rank_key, recording_distances = umeerj_resolve_recording_distances(
        distances_by_model=distances_by_model,
        encoder=encoder,
        layer=layer,
        dataset=dataset,
        strategy=strategy,
        phone_class=phone_class,
        alignment=alignment,
        rank=rank,
    )

    scores_by_rater = scores_by_dataset[dataset]

    config = {
        "encoder": encoder,
        "layer": layer,
        "dataset": dataset,
        "strategy": strategy,
        "phone_class": phone_class,
        "alignment": alignment,
        "rank": rank_key,
        "metric": metric,
        "phone_aggregation": phone_aggregation,
        "orientation": "raw_distance",
        "expected_good_sign": "negative",
    }

    records = umeerj_collect_distance_rater_records_clean(
        recording_distances=recording_distances,
        scores_by_rater=scores_by_rater,
        metric=metric,
        phone_aggregation=phone_aggregation,
        trim_ratio=trim_ratio,
        speaker_by_recording=speaker_by_recording,
        silence_by_recording=silence_by_recording,
        silence_key=silence_key,
        config=config,
    )

    model_eval = umeerj_evaluate_records_by_rater_clean(
        records=records,
        controls=controls,
        min_n=min_n,
        average_method=average_method,
    )

    human_eval = umeerj_human_benchmark_from_records_clean(
        records=records,
        item_keys=("recording_id",),
        min_n=min_n,
        average_method=average_method,
    )

    auxiliary = umeerj_auxiliary_associations_by_rater_clean(
        records=records,
        aux_names=("n_common", "silence"),
        min_n=min_n,
        average_method=average_method,
    )

    diagnostics = build_record_diagnostics(records)

    result = {
        "config": config,
        "model_vs_rater": model_eval,
        "human_benchmark": human_eval,
        "auxiliary": auxiliary,
        "diagnostics": diagnostics,
    }

    # Backwards-friendly aliases.
    result["macro"] = model_eval["macro"]
    result["per_rater"] = model_eval["per_rater"]

    if bootstrap:
        result["speaker_cluster_bootstrap"] = umeerj_speaker_cluster_bootstrap_r_clean(
            records=records,
            cluster_key="speaker_id",
            rater_key="rater",
            score_key="score",
            value_key="value",
            n_boot=n_boot,
            seed=seed,
            min_n=min_n,
            average_method=average_method,
        )

    if return_records:
        result["records"] = records

    return result


###############################################################################
# Grid evaluator
###############################################################################

def umeerj_evaluate_distance_grid_clean(
    distances_by_model: dict[str, dict[int, dict[str, dict]]],
    scores_by_dataset: dict[str, dict[str, dict[str, int]]],
    encoder: str,
    layer: int,
    dataset: str,
    strategies: Iterable[str] | None = None,
    phone_classes: Iterable[str] | None = None,
    alignments: Iterable[str] | None = None,
    ranks: Iterable[Any] | None = None,
    metrics: Iterable[str] | None = None,
    phone_aggregation: str = "mean",
    trim_ratio: float = 0.1,
    speaker_by_recording: dict[str, str] | None = None,
    silence_by_recording: dict[str, Any] | None = None,
    silence_key: str = "silence_ratio",
    controls: tuple[str, ...] | None = ("n_common", "silence"),
    min_n: int = 10,
    average_method: str = "arithmetic",
    bootstrap: bool = False,
    n_boot: int = 10000,
    seed: int = 0,
    return_records: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a full nested UME-ERJ grid for one encoder/layer/dataset.

    Returns:
        results[strategy][phone_class][alignment][rank][metric]
    """
    distance_grid = distances_by_model[encoder][layer][dataset]

    if strategies is None:
        strategies = sorted(distance_grid.keys(), key=str)

    results: dict[str, Any] = {}

    for strategy in strategies:
        if strategy not in distance_grid:
            continue

        results.setdefault(strategy, {})

        strategy_data = distance_grid[strategy]

        pc_iter = phone_classes
        if pc_iter is None:
            pc_iter = sorted(strategy_data.keys(), key=str)

        for phone_class in pc_iter:
            if phone_class not in strategy_data:
                continue

            results[strategy].setdefault(phone_class, {})

            pc_data = strategy_data[phone_class]

            align_iter = alignments
            if align_iter is None:
                align_iter = sorted(pc_data.keys(), key=str)

            for alignment in align_iter:
                if alignment not in pc_data:
                    continue

                results[strategy][phone_class].setdefault(alignment, {})

                align_data = pc_data[alignment]

                rank_iter = ranks
                if rank_iter is None:
                    rank_iter = sorted(align_data.keys(), key=str)

                for rank in rank_iter:
                    rank_key = None

                    for candidate in [rank, str(rank)]:
                        if candidate in align_data:
                            rank_key = candidate
                            break

                    if rank_key is None:
                        try:
                            candidate = int(rank)
                            if candidate in align_data:
                                rank_key = candidate
                        except Exception:
                            pass

                    if rank_key is None:
                        continue

                    recording_distances = align_data[rank_key]

                    metric_iter = metrics
                    if metric_iter is None:
                        metric_iter = umeerj_get_available_distance_metrics(
                            recording_distances
                        )

                    results[strategy][phone_class][alignment].setdefault(rank_key, {})

                    for metric in metric_iter:
                        metric = str(metric)

                        print(
                            f"Evaluating UME-ERJ raw distances | "
                            f"encoder={encoder} | layer={layer} | dataset={dataset} | "
                            f"strategy={strategy} | phone_class={phone_class} | "
                            f"alignment={alignment} | rank={rank_key} | metric={metric}"
                        )

                        result = umeerj_evaluate_one_config_clean(
                            distances_by_model=distances_by_model,
                            scores_by_dataset=scores_by_dataset,
                            encoder=encoder,
                            layer=layer,
                            dataset=dataset,
                            strategy=strategy,
                            phone_class=phone_class,
                            alignment=alignment,
                            rank=rank_key,
                            metric=metric,
                            phone_aggregation=phone_aggregation,
                            trim_ratio=trim_ratio,
                            speaker_by_recording=speaker_by_recording,
                            silence_by_recording=silence_by_recording,
                            silence_key=silence_key,
                            controls=controls,
                            min_n=min_n,
                            average_method=average_method,
                            bootstrap=bootstrap,
                            n_boot=n_boot,
                            seed=seed,
                            return_records=return_records,
                        )

                        results[strategy][phone_class][alignment][rank_key][metric] = result

    return results


###############################################################################
# Flattening utilities
###############################################################################

def umeerj_clean_result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten one config result to one summary row.
    """
    config = result.get("config", {})
    macro = result.get("model_vs_rater", {}).get("macro", {})
    human = result.get("human_benchmark", {}).get("macro", {})
    diagnostics = result.get("diagnostics", {})

    row = {}

    row.update(config)

    for key, value in macro.items():
        row[f"model_{key}"] = value

    for key, value in human.items():
        row[f"human_{key}"] = value

    for key in [
        "n_records",
        "n_unique_recordings",
        "n_unique_speakers",
        "n_raters",
        "finite_score",
        "finite_value",
        "finite_n_common",
        "finite_silence",
        "unique_score",
        "unique_value",
        "unique_n_common",
        "unique_silence",
    ]:
        if key in diagnostics:
            row[f"diag_{key}"] = diagnostics[key]

    if "speaker_cluster_bootstrap" in result:
        boot = result["speaker_cluster_bootstrap"]

        model_boot = boot.get("model_macro_pearson_r", {})
        human_boot = boot.get("human_macro_pearson_r", {})
        strength_boot = boot.get("distance_strength_macro_pearson_r", {})

        row["boot_model_pearson_point"] = model_boot.get("point", np.nan)
        row["boot_model_pearson_ci_lower"] = model_boot.get("ci_lower", np.nan)
        row["boot_model_pearson_ci_upper"] = model_boot.get("ci_upper", np.nan)
        row["boot_model_pearson_p_negative_vs_zero"] = model_boot.get(
            "p_negative_vs_zero",
            np.nan,
        )

        row["boot_human_pearson_point"] = human_boot.get("point", np.nan)
        row["boot_human_pearson_ci_lower"] = human_boot.get("ci_lower", np.nan)
        row["boot_human_pearson_ci_upper"] = human_boot.get("ci_upper", np.nan)

        row["boot_distance_strength_point"] = strength_boot.get("point", np.nan)
        row["boot_distance_strength_ci_lower"] = strength_boot.get("ci_lower", np.nan)
        row["boot_distance_strength_ci_upper"] = strength_boot.get("ci_upper", np.nan)
        row["boot_distance_strength_p_positive_vs_zero"] = strength_boot.get(
            "p_positive_vs_zero",
            np.nan,
        )

    return row


def umeerj_clean_grid_to_rows(eval_grid: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten:
        results[strategy][phone_class][alignment][rank][metric]

    to list of rows.
    """
    rows = []

    for strategy, strategy_data in eval_grid.items():
        for phone_class, pc_data in strategy_data.items():
            for alignment, align_data in pc_data.items():
                for rank, rank_data in align_data.items():
                    for metric, result in rank_data.items():
                        row = umeerj_clean_result_to_row(result)

                        row.setdefault("strategy", strategy)
                        row.setdefault("phone_class", phone_class)
                        row.setdefault("alignment", alignment)
                        row.setdefault("rank", rank)
                        row.setdefault("metric", metric)

                        rows.append(row)

    return rows


def umeerj_clean_grid_to_summary_df(eval_grid: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(umeerj_clean_grid_to_rows(eval_grid))


def umeerj_per_rater_rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten per-rater results for one config.
    """
    config = result.get("config", {})
    per_rater = result.get("model_vs_rater", {}).get("per_rater", {})

    rows = []

    for rater, item in per_rater.items():
        row = dict(config)
        row["rater"] = rater
        row.update(item)
        rows.append(row)

    return rows


def umeerj_per_pair_rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten human-human per-pair results for one config.
    """
    config = result.get("config", {})
    per_pair = result.get("human_benchmark", {}).get("per_pair", {})

    rows = []

    for pair, item in per_pair.items():
        row = dict(config)
        row["pair"] = pair
        row.update(item)
        rows.append(row)

    return rows


def _unique_keep_order(items):
    out = []

    for item in items:
        if item not in out:
            out.append(item)

    return out


def _tie_info(values):
    """
    Precompute sorting/tie groups for weighted midranks.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return {
            "n": 0,
            "order": np.asarray([], dtype=int),
            "starts": np.asarray([], dtype=int),
            "ends": np.asarray([], dtype=int),
        }

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    if sorted_values.size == 1:
        starts = np.asarray([0], dtype=int)
        ends = np.asarray([1], dtype=int)
    else:
        change = np.r_[
            True,
            sorted_values[1:] != sorted_values[:-1],
        ]

        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], sorted_values.size]

    return {
        "n": int(values.size),
        "order": order,
        "starts": starts,
        "ends": ends,
    }


def _weighted_midranks_from_tie_info(weights, tie_info):
    """
    Exact midranks for an expanded bootstrap sample represented by weights.

    If an observation has bootstrap weight 3, it is treated as if it appeared
    3 times in the bootstrap sample.

    This gives exact Spearman ranks under cluster resampling with replacement.
    """
    weights = np.asarray(weights, dtype=float)

    n = tie_info["n"]
    order = tie_info["order"]
    starts = tie_info["starts"]
    ends = tie_info["ends"]

    ranks_sorted = np.full(n, np.nan, dtype=float)

    if n == 0:
        return ranks_sorted, 0.0

    sorted_weights = weights[order]

    cumulative_weight = 0.0

    for start, end in zip(starts, ends):
        group_weight = float(np.sum(sorted_weights[start:end]))

        if group_weight > 0:
            # 1-based midrank of the expanded tie block.
            midrank = cumulative_weight + (group_weight + 1.0) / 2.0
            ranks_sorted[start:end] = midrank
            cumulative_weight += group_weight
        else:
            ranks_sorted[start:end] = np.nan

    ranks = np.full(n, np.nan, dtype=float)
    ranks[order] = ranks_sorted

    return ranks, cumulative_weight


def _weighted_pearson(x, y, weights, min_n: int = 10):
    """
    Weighted Pearson correlation.

    For integer bootstrap weights, this equals Pearson on the expanded sample.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(weights)
        & (weights > 0)
    )

    x = x[mask]
    y = y[mask]
    weights = weights[mask]

    total_weight = float(np.sum(weights))

    if total_weight < min_n:
        return np.nan

    mean_x = float(np.sum(weights * x) / total_weight)
    mean_y = float(np.sum(weights * y) / total_weight)

    xc = x - mean_x
    yc = y - mean_y

    cov_xy = float(np.sum(weights * xc * yc))
    var_x = float(np.sum(weights * xc * xc))
    var_y = float(np.sum(weights * yc * yc))

    denom = np.sqrt(var_x * var_y)

    if not np.isfinite(denom) or denom <= 0:
        return np.nan

    return float(cov_xy / denom)


def _weighted_spearman_precomputed(
    x,
    y,
    weights,
    tie_info_x,
    tie_info_y,
    min_n: int = 10,
):
    """
    Exact weighted Spearman correlation.

    Spearman is Pearson correlation between ranks.
    Here the ranks are weighted midranks corresponding to the expanded
    bootstrap sample.
    """
    weights = np.asarray(weights, dtype=float)

    rank_x, total_x = _weighted_midranks_from_tie_info(
        weights=weights,
        tie_info=tie_info_x,
    )

    rank_y, total_y = _weighted_midranks_from_tie_info(
        weights=weights,
        tie_info=tie_info_y,
    )

    total_weight = min(total_x, total_y)

    if total_weight < min_n:
        return np.nan

    return _weighted_pearson(
        rank_x,
        rank_y,
        weights,
        min_n=min_n,
    )


def _macro_values(values, average_method: str = "arithmetic"):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    if average_method == "arithmetic":
        return float(np.mean(arr))

    if average_method == "fisher":
        arr = np.clip(arr, -0.999999, 0.999999)
        return float(np.tanh(np.mean(np.arctanh(arr))))

    raise ValueError("average_method must be 'arithmetic' or 'fisher'.")


def umeerj_fast_speaker_cluster_bootstrap_spearman(
    records,
    cluster_key: str = "speaker_id",
    item_key: str = "recording_id",
    rater_key: str = "rater",
    score_key: str = "score",
    value_key: str = "value",
    n_boot: int = 10000,
    seed: int = 0,
    min_n: int = 10,
    average_method: str = "arithmetic",
    include_human: bool = True,
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    Fast speaker/recording-cluster bootstrap for macro Spearman rho.

    Main model statistic:
        for each rater:
            Spearman(score, raw_distance)

        then average across raters.

    Human benchmark:
        for every rater pair:
            Spearman(score_rater_a, score_rater_b)

        then average across pairs.

    Raw-distance convention:
        good model-distance association is negative.

    Notes
    -----
    This is exact for cluster bootstrap with replacement in the sense that
    duplicated clusters are represented as integer weights and Spearman
    midranks are recomputed for the weighted expanded sample.
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return {
            "model_macro_spearman_rho": _bootstrap_ci([], np.nan),
            "human_macro_spearman_rho": _bootstrap_ci([], np.nan),
            "distance_strength_macro_spearman_rho": _bootstrap_ci([], np.nan),
            "diagnostics": {
                "n_records": 0,
                "n_clusters": 0,
                "fast": True,
                "metric": "spearman",
            },
        }

    required = [
        cluster_key,
        item_key,
        rater_key,
        score_key,
        value_key,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns for Spearman bootstrap: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df[cluster_key] = df[cluster_key].astype(str)
    df[item_key] = df[item_key].astype(str)
    df[rater_key] = df[rater_key].astype(str)

    df[score_key] = pd.to_numeric(df[score_key], errors="coerce")
    df[value_key] = pd.to_numeric(df[value_key], errors="coerce")

    finite = (
        np.isfinite(df[score_key].to_numpy(dtype=float))
        & np.isfinite(df[value_key].to_numpy(dtype=float))
    )

    df = df.loc[finite].copy()

    cluster_ids = sorted(df[cluster_key].unique(), key=str)
    raters = sorted(df[rater_key].unique(), key=str)

    n_clusters = len(cluster_ids)

    if n_clusters == 0 or len(raters) == 0:
        return {
            "model_macro_spearman_rho": _bootstrap_ci([], np.nan),
            "human_macro_spearman_rho": _bootstrap_ci([], np.nan),
            "distance_strength_macro_spearman_rho": _bootstrap_ci([], np.nan),
            "diagnostics": {
                "n_records": int(len(df)),
                "n_clusters": int(n_clusters),
                "n_raters": int(len(raters)),
                "fast": True,
                "metric": "spearman",
            },
        }

    cluster_to_idx = {
        cluster: i
        for i, cluster in enumerate(cluster_ids)
    }

    # ------------------------------------------------------------------
    # Precompute model-vs-rater groups.
    # ------------------------------------------------------------------
    model_groups = []

    for rater, g in df.groupby(rater_key, sort=True):
        cluster_idx = np.asarray(
            [
                cluster_to_idx[str(c)]
                for c in g[cluster_key].tolist()
            ],
            dtype=int,
        )

        scores = g[score_key].to_numpy(dtype=float)
        values = g[value_key].to_numpy(dtype=float)

        model_groups.append(
            {
                "rater": str(rater),
                "cluster_idx": cluster_idx,
                "score": scores,
                "value": values,
                "tie_score": _tie_info(scores),
                "tie_value": _tie_info(values),
            }
        )

    # ------------------------------------------------------------------
    # Precompute human-human pair groups.
    # ------------------------------------------------------------------
    human_groups = []

    if include_human:
        rating_cols = _unique_keep_order(
            [
                cluster_key,
                item_key,
                rater_key,
                score_key,
            ]
        )

        group_cols = _unique_keep_order(
            [
                cluster_key,
                item_key,
                rater_key,
            ]
        )

        rating_df = (
            df[rating_cols]
            .groupby(group_cols, as_index=False)
            .agg({score_key: "mean"})
        )

        merge_cols = _unique_keep_order(
            [
                cluster_key,
                item_key,
            ]
        )

        for rater_a, rater_b in combinations(raters, 2):
            da = rating_df[
                rating_df[rater_key].astype(str) == str(rater_a)
            ][merge_cols + [score_key]].copy()

            db = rating_df[
                rating_df[rater_key].astype(str) == str(rater_b)
            ][merge_cols + [score_key]].copy()

            pair_df = da.merge(
                db,
                on=merge_cols,
                how="inner",
                suffixes=("_a", "_b"),
            )

            if len(pair_df) == 0:
                human_groups.append(
                    {
                        "pair": f"{rater_a}__{rater_b}",
                        "cluster_idx": np.asarray([], dtype=int),
                        "score_a": np.asarray([], dtype=float),
                        "score_b": np.asarray([], dtype=float),
                        "tie_a": _tie_info([]),
                        "tie_b": _tie_info([]),
                    }
                )
                continue

            cluster_idx = np.asarray(
                [
                    cluster_to_idx[str(c)]
                    for c in pair_df[cluster_key].tolist()
                ],
                dtype=int,
            )

            score_a = pair_df[f"{score_key}_a"].to_numpy(dtype=float)
            score_b = pair_df[f"{score_key}_b"].to_numpy(dtype=float)

            finite_pair = np.isfinite(score_a) & np.isfinite(score_b)

            cluster_idx = cluster_idx[finite_pair]
            score_a = score_a[finite_pair]
            score_b = score_b[finite_pair]

            human_groups.append(
                {
                    "pair": f"{rater_a}__{rater_b}",
                    "cluster_idx": cluster_idx,
                    "score_a": score_a,
                    "score_b": score_b,
                    "tie_a": _tie_info(score_a),
                    "tie_b": _tie_info(score_b),
                }
            )

    # ------------------------------------------------------------------
    # Point estimates.
    # ------------------------------------------------------------------
    point_model_rhos = []

    for group in model_groups:
        weights = np.ones(len(group["score"]), dtype=float)

        rho = _weighted_spearman_precomputed(
            x=group["score"],
            y=group["value"],
            weights=weights,
            tie_info_x=group["tie_score"],
            tie_info_y=group["tie_value"],
            min_n=min_n,
        )

        if np.isfinite(rho):
            point_model_rhos.append(float(rho))

    point_model_rho = _macro_values(
        point_model_rhos,
        average_method=average_method,
    )

    point_distance_strength = (
        -point_model_rho
        if np.isfinite(point_model_rho)
        else np.nan
    )

    point_human_rhos = []

    if include_human:
        for group in human_groups:
            weights = np.ones(len(group["score_a"]), dtype=float)

            rho = _weighted_spearman_precomputed(
                x=group["score_a"],
                y=group["score_b"],
                weights=weights,
                tie_info_x=group["tie_a"],
                tie_info_y=group["tie_b"],
                min_n=min_n,
            )

            if np.isfinite(rho):
                point_human_rhos.append(float(rho))

    point_human_rho = _macro_values(
        point_human_rhos,
        average_method=average_method,
    )

    # ------------------------------------------------------------------
    # Bootstrap.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    probs = np.full(
        n_clusters,
        1.0 / n_clusters,
        dtype=float,
    )

    model_boot = []
    strength_boot = []
    human_boot = []

    n_done = 0

    while n_done < n_boot:
        current_batch = min(
            batch_size,
            n_boot - n_done,
        )

        count_batch = rng.multinomial(
            n=n_clusters,
            pvals=probs,
            size=current_batch,
        ).astype(float)

        for counts in count_batch:
            # Model-vs-rater macro Spearman.
            rhos = []

            for group in model_groups:
                weights = counts[group["cluster_idx"]]

                rho = _weighted_spearman_precomputed(
                    x=group["score"],
                    y=group["value"],
                    weights=weights,
                    tie_info_x=group["tie_score"],
                    tie_info_y=group["tie_value"],
                    min_n=min_n,
                )

                if np.isfinite(rho):
                    rhos.append(float(rho))

            macro_model_rho = _macro_values(
                rhos,
                average_method=average_method,
            )

            if np.isfinite(macro_model_rho):
                model_boot.append(float(macro_model_rho))
                strength_boot.append(float(-macro_model_rho))

            # Human-human macro Spearman.
            if include_human:
                pair_rhos = []

                for group in human_groups:
                    weights = counts[group["cluster_idx"]]

                    rho = _weighted_spearman_precomputed(
                        x=group["score_a"],
                        y=group["score_b"],
                        weights=weights,
                        tie_info_x=group["tie_a"],
                        tie_info_y=group["tie_b"],
                        min_n=min_n,
                    )

                    if np.isfinite(rho):
                        pair_rhos.append(float(rho))

                macro_human_rho = _macro_values(
                    pair_rhos,
                    average_method=average_method,
                )

                if np.isfinite(macro_human_rho):
                    human_boot.append(float(macro_human_rho))

        n_done += current_batch

    model_arr = finite_array(model_boot)
    strength_arr = finite_array(strength_boot)

    if model_arr.size > 0:
        # One-sided p-value for expected negative raw-distance correlation.
        p_negative_vs_zero = float(
            (1.0 + np.sum(model_arr >= 0.0)) / (1.0 + model_arr.size)
        )
    else:
        p_negative_vs_zero = np.nan

    if strength_arr.size > 0:
        # Equivalent p-value for -rho > 0.
        p_strength_positive_vs_zero = float(
            (1.0 + np.sum(strength_arr <= 0.0)) / (1.0 + strength_arr.size)
        )
    else:
        p_strength_positive_vs_zero = np.nan

    return {
        "model_macro_spearman_rho": {
            **_bootstrap_ci(model_boot, point_model_rho),
            "expected_good_sign": "negative",
            "p_negative_vs_zero": p_negative_vs_zero,
        },
        "human_macro_spearman_rho": {
            **_bootstrap_ci(human_boot, point_human_rho),
            "expected_good_sign": "positive",
        },
        "distance_strength_macro_spearman_rho": {
            **_bootstrap_ci(strength_boot, point_distance_strength),
            "definition": "-model_macro_spearman_rho",
            "expected_good_sign": "positive",
            "p_positive_vs_zero": p_strength_positive_vs_zero,
        },
        "diagnostics": {
            "n_records": int(len(df)),
            "n_clusters": int(n_clusters),
            "n_raters": int(len(model_groups)),
            "n_pairs": int(len(human_groups)),
            "cluster_key": cluster_key,
            "item_key": item_key,
            "n_boot": int(n_boot),
            "min_n": int(min_n),
            "average_method": average_method,
            "include_human": bool(include_human),
            "fast": True,
            "metric": "spearman",
        },
    }