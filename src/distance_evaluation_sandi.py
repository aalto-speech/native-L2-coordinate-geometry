"""SANDI distance-evaluation utilities.

These functions evaluate associations between part-level SANDI distance
measurements and CEFR levels.  UME-ERJ's recording-level, rater-aware
evaluation lives separately in :mod:`distance_evaluation_umeerj`.
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import (
    spearmanr,
    kendalltau,
    pearsonr,
    rankdata,
)


from config import (
    SANDI_CEFR_DICT,
)

from utils import (
    load_pickle,
    make_cosine_noise_suffix,
    save_pickle,
)


def level_to_score(level):
    """
    Convert level key to numeric CEFR score.
    """
    if isinstance(level, (int, float, np.integer, np.floating)):
        return float(level)

    level_str = str(level).strip().lower()


    if level_str in SANDI_CEFR_DICT:
        return SANDI_CEFR_DICT[level_str]

    return float(level_str)


def distance_to_scalar(value, aggregation="mean"):
    """
    Convert scalar or per-phone/per-diphone distance vector
    to one recording-level scalar.
    """
    arr = np.asarray(value, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    if aggregation == "mean":
        return float(np.mean(arr))

    if aggregation == "median":
        return float(np.median(arr))

    raise ValueError(f"Unknown aggregation: {aggregation}")


def safe_corr(y, x, method="spearman"):
    """
    Safe correlation between CEFR score y and system score x.

    For raw distance/error scores, negative correlation is expected.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(y) & np.isfinite(x)

    y = y[mask]
    x = x[mask]

    if len(y) < 3:
        return {
            "n": int(len(y)),
            "stat": np.nan,
            "p_value": np.nan,
        }

    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {
            "n": int(len(y)),
            "stat": np.nan,
            "p_value": np.nan,
        }

    if method == "spearman":
        stat, p_value = spearmanr(y, x)

    elif method == "kendall":
        stat, p_value = kendalltau(y, x)

    elif method == "pearson":
        stat, p_value = pearsonr(y, x)

    else:
        raise ValueError(
            "method must be 'spearman', 'kendall', or 'pearson'."
        )

    return {
        "n": int(len(y)),
        "stat": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
    }


def fisher_average_correlations(rhos):
    """
    Fisher-average correlations over parts.
    Keeps the sign, so negative macro rho means expected association
    for distance/error scores.
    """
    rhos = np.asarray(rhos, dtype=float)
    rhos = rhos[np.isfinite(rhos)]

    if len(rhos) == 0:
        return np.nan

    rhos = np.clip(rhos, -0.999999, 0.999999)

    z = np.arctanh(rhos)

    return float(np.tanh(np.mean(z)))



def evaluate_auxiliary_associations(records):
    """
    Evaluate how auxiliary variables correlate with CEFR.
    """
    aux_specs = {
        "n_common": lambda r: r.get("n_common", np.nan),
        "log_n_common": lambda r: (
            np.log1p(r.get("n_common", np.nan))
            if np.isfinite(r.get("n_common", np.nan))
            else np.nan
        ),
        "silence": lambda r: r.get("silence", np.nan),
    }

    results = {}

    for name, getter in aux_specs.items():
        aux_records = []

        for r in records:
            value = getter(r)

            if not np.isfinite(value):
                continue

            rr = dict(r)
            rr["value"] = float(value)
            aux_records.append(rr)

        results[name] = evaluate_association(aux_records)

    return results

def evaluate_records_correlation(records):
    """
    Evaluate CEFR association for one record list.

    For raw distance/error values:
        negative correlations are expected.
    """
    scores = [
        r["score"]
        for r in records
    ]

    values = [
        r["value"]
        for r in records
    ]

    spearman = safe_corr(
        scores,
        values,
        method="spearman",
    )

    kendall = safe_corr(
        scores,
        values,
        method="kendall",
    )

    pearson = safe_corr(
        scores,
        values,
        method="pearson",
    )

    return {
        "n": spearman["n"],

        "spearman_rho": spearman["stat"],
        "spearman_p_value": spearman["p_value"],

        "kendall_tau": kendall["stat"],
        "kendall_p_value": kendall["p_value"],

        "pearson_r": pearson["stat"],
        "pearson_p_value": pearson["p_value"],
    }


def evaluate_by_part(records):
    """
    Compute correlations separately for each part.
    """
    parts = sorted(
        set(
            r["part"]
            for r in records
        ),
        key=lambda x: str(x),
    )

    result = {}

    for part in parts:

        part_records = [
            r
            for r in records
            if r["part"] == part
        ]

        result[part] = evaluate_records_correlation(
            part_records
        )

    return result


def macro_from_part_results(part_results):
    """
    Macro-average correlations over parts.
    """
    spearmans = []
    kendalls = []
    pearsons = []

    n_total = 0

    for part, res in part_results.items():

        n_total += res["n"]

        if np.isfinite(res["spearman_rho"]):
            spearmans.append(res["spearman_rho"])

        if np.isfinite(res["kendall_tau"]):
            kendalls.append(res["kendall_tau"])

        if np.isfinite(res["pearson_r"]):
            pearsons.append(res["pearson_r"])

    return {
        "n_total": int(n_total),
        "n_parts": int(len(spearmans)),

        "macro_spearman_rho": fisher_average_correlations(spearmans),
        "macro_kendall_tau": fisher_average_correlations(kendalls),
        "macro_pearson_r": fisher_average_correlations(pearsons),
    }


def evaluate_association(records):
    """
    Full correlation evaluation:
        pooled + per-part + macro.
    """
    pooled = evaluate_records_correlation(records)

    by_part = evaluate_by_part(records)

    macro = macro_from_part_results(by_part)

    return {
        "pooled": pooled,
        "by_part": by_part,
        "macro": macro,
    }


def ridge_fusion_train_testV2(
    train_records,
    test_records,
    feature_cols=("distance", "log_n_common", "silence"),
    alpha=1.0,
    include_part=False,
):
    """
    Supervised linear fusion baseline.

    Fits ridge regression to predict numeric CEFR score from:
        - native-reference distance
        - log number of common phone classes
        - silence ratio
        - optional part indicators

    Important:
        This treats CEFR as numeric during training.
        Evaluation can still be rank-based using Spearman.
    """
    train_df = prepare_fusion_dataframe(train_records)
    test_df = prepare_fusion_dataframe(test_records)

    train_df = train_df[np.isfinite(train_df["score"])].copy()
    test_df = test_df[np.isfinite(test_df["score"])].copy()

    feature_names = list(feature_cols)

    # Impute and standardize numeric features using training statistics only.
    for col in feature_cols:
        train_values = train_df[col].to_numpy(dtype=float)

        mean_value = float(np.nanmean(train_values))
        std_value = float(np.nanstd(train_values))

        if not np.isfinite(mean_value):
            mean_value = 0.0

        if not np.isfinite(std_value) or std_value <= 0:
            std_value = 1.0

        train_df[col] = train_df[col].fillna(mean_value)
        test_df[col] = test_df[col].fillna(mean_value)

        train_df[col] = (train_df[col] - mean_value) / std_value
        test_df[col] = (test_df[col] - mean_value) / std_value

    X_train_parts = [
        train_df[list(feature_cols)].to_numpy(dtype=float)
    ]

    X_test_parts = [
        test_df[list(feature_cols)].to_numpy(dtype=float)
    ]

    if include_part:
        train_parts = train_df["part"].astype(str)
        test_parts = test_df["part"].astype(str)

        part_levels = sorted(train_parts.unique())

        # Drop first part to avoid collinearity with the intercept.
        for part in part_levels[1:]:
            feature_names.append(f"part={part}")

            X_train_parts.append(
                (train_parts == part).astype(float).to_numpy()[:, None]
            )

            X_test_parts.append(
                (test_parts == part).astype(float).to_numpy()[:, None]
            )

    X_train = np.concatenate(X_train_parts, axis=1)
    X_test = np.concatenate(X_test_parts, axis=1)

    # Add intercept.
    X_train = np.column_stack(
        [
            np.ones(len(X_train)),
            X_train,
        ]
    )

    X_test = np.column_stack(
        [
            np.ones(len(X_test)),
            X_test,
        ]
    )

    coef_names = ["intercept"] + feature_names

    y_train = train_df["score"].to_numpy(dtype=float)
    y_test = test_df["score"].to_numpy(dtype=float)

    # Do not penalize intercept.
    penalty = np.eye(X_train.shape[1])
    penalty[0, 0] = 0.0

    beta = np.linalg.solve(
        X_train.T @ X_train + alpha * penalty,
        X_train.T @ y_train,
    )

    y_pred = X_test @ beta

    spearman = safe_corr(
        y_test,
        y_pred,
        method="spearman",
    )

    pearson = safe_corr(
        y_test,
        y_pred,
        method="pearson",
    )

    kendall = safe_corr(
        y_test,
        y_pred,
        method="kendall",
    )

    coefficients = {
        name: float(value)
        for name, value in zip(coef_names, beta)
    }

    return {
        "coefficients": coefficients,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "spearman": spearman,
        "pearson": pearson,
        "kendall": kendall,
        "y_test": y_test,
        "y_pred": y_pred,
    }



def add_silence_to_records(
    records,
    silence_by_recording,
    silence_key="silence",
    value_key="silence_ratio",
):
    """
    Attach a scalar silence value to records.

    silence_by_recording may map recording_id -> dict,
    in which case value_key is extracted from the dict.
    """
    silence_norm = {}

    for key, value in silence_by_recording.items():
        if isinstance(key, tuple) and len(key) == 2:
            silence_norm[(str(key[0]), str(key[1]))] = value
        else:
            silence_norm[str(key)] = value

    output = []

    for r in records:
        item = dict(r)

        recording_id = str(item["recording_id"])
        part = str(item.get("part", ""))

        raw = np.nan

        if (part, recording_id) in silence_norm:
            raw = silence_norm[(part, recording_id)]
        elif recording_id in silence_norm:
            raw = silence_norm[recording_id]

        # Extract scalar from dict if needed.
        if isinstance(raw, dict):
            value = raw.get(value_key, np.nan)
        else:
            value = raw

        try:
            value = float(value)
        except Exception:
            value = np.nan

        item[silence_key] = value
        output.append(item)

    return output



def prepare_fusion_dataframe(records):
    """
    Convert records to a DataFrame for feature fusion.
    """
    df = pd.DataFrame(records).copy()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    elif "value" in df.columns:
        df["distance"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        raise KeyError("Neither 'distance' nor 'value' found in records.")

    if "n_common" in df.columns:
        df["n_common"] = pd.to_numeric(df["n_common"], errors="coerce")
    else:
        df["n_common"] = np.nan

    df["log_n_common"] = np.log1p(df["n_common"].clip(lower=0))

    if "silence" in df.columns:
        df["silence"] = pd.to_numeric(df["silence"], errors="coerce")
    else:
        df["silence"] = np.nan

    return df


def ridge_fusion_train_test(
    train_records,
    test_records,
    feature_cols=("distance", "log_n_common", "silence"),
    alpha=1.0,
    include_part=False,
):
    """
    Fit a simple ridge regression on train records and evaluate on test records.

    The target is CEFR score.
    Distance is used as a raw feature, so the model can learn its sign.
    """
    train_df = prepare_fusion_dataframe(train_records)
    test_df = prepare_fusion_dataframe(test_records)

    # Keep rows with finite target.
    train_df = train_df[np.isfinite(train_df["score"])].copy()
    test_df = test_df[np.isfinite(test_df["score"])].copy()

    # Impute and standardize numeric features using train statistics only.
    for col in feature_cols:
        train_values = train_df[col].to_numpy(dtype=float)

        mean_value = float(np.nanmean(train_values))
        std_value = float(np.nanstd(train_values))

        if not np.isfinite(mean_value):
            mean_value = 0.0

        if not np.isfinite(std_value) or std_value <= 0:
            std_value = 1.0

        train_df[col] = train_df[col].fillna(mean_value)
        test_df[col] = test_df[col].fillna(mean_value)

        train_df[col] = (train_df[col] - mean_value) / std_value
        test_df[col] = (test_df[col] - mean_value) / std_value

    X_train_parts = [
        train_df[list(feature_cols)].to_numpy(dtype=float)
    ]

    X_test_parts = [
        test_df[list(feature_cols)].to_numpy(dtype=float)
    ]

    if include_part:
        train_parts = train_df["part"].astype(str)
        test_parts = test_df["part"].astype(str)

        part_levels = sorted(train_parts.unique())

        # Drop first part as reference.
        for part in part_levels[1:]:
            X_train_parts.append(
                (train_parts == part).astype(float).to_numpy()[:, None]
            )
            X_test_parts.append(
                (test_parts == part).astype(float).to_numpy()[:, None]
            )

    X_train = np.concatenate(
        X_train_parts,
        axis=1,
    )

    X_test = np.concatenate(
        X_test_parts,
        axis=1,
    )

    # Add intercept.
    X_train = np.column_stack(
        [
            np.ones(len(X_train)),
            X_train,
        ]
    )

    X_test = np.column_stack(
        [
            np.ones(len(X_test)),
            X_test,
        ]
    )

    y_train = train_df["score"].to_numpy(dtype=float)
    y_test = test_df["score"].to_numpy(dtype=float)

    # Do not penalize intercept.
    penalty = np.eye(X_train.shape[1])
    penalty[0, 0] = 0.0

    beta = np.linalg.solve(
        X_train.T @ X_train + alpha * penalty,
        X_train.T @ y_train,
    )

    y_pred = X_test @ beta

    spearman = safe_corr(
        y_test,
        y_pred,
        method="spearman",
    )

    pearson = safe_corr(
        y_test,
        y_pred,
        method="pearson",
    )

    kendall = safe_corr(
        y_test,
        y_pred,
        method="kendall",
    )

    return {
        "beta": beta,
        "feature_cols": ("intercept",) + tuple(feature_cols),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "spearman": spearman,
        "pearson": pearson,
        "kendall": kendall,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def collect_records_from_nested_distances(
    distances,
    encoder,
    dataset,
    layer,
    strategy,
    phone_class,
    alignment,
    rank,
    metric,
    max_i=100000,
    aggregation="mean",
):
    """
    Collect recording-level records from nested distance dictionary.

    value = raw distance/error score.

    Therefore, for distance/error measures:
        Spearman(CEFR, value) should be negative.
    """
    ds = (
        distances[encoder]
        [layer]
        [dataset]
        [strategy]
        [phone_class]
        [alignment]
        [rank]
    )

    records = []

    for part in ds:

        for level in ds[part]:

            if metric not in ds[part][level]:
                continue

            score = level_to_score(level)

            recording_distances = ds[part][level][metric]

            recording_ids = list(recording_distances.keys())

            for i, recording_id in enumerate(recording_ids):

                if i >= max_i:
                    break

                value = distance_to_scalar(
                    recording_distances[recording_id],
                    aggregation=aggregation,
                )

                if not np.isfinite(value):
                    continue

                n_common = np.nan

                if "common_phones" in ds[part][level]:
                    common_dict = ds[part][level]["common_phones"]

                    if recording_id in common_dict:
                        n_common = len(common_dict[recording_id])

                records.append(
                    {
                        "recording_id": recording_id,
                        "part": part,
                        "level": level,
                        "score": float(score),
                        "value": float(value),
                        "n_common": n_common,
                        "metric": metric,
                        "phone_class": phone_class,
                    }
                )

    return records



def bootstrap_ci(
    records,
    stat_name="macro_spearman_rho",
    n_boot=5000,
    seed=0,
    stratify_by_part=True,
):
    """
    Bootstrap CI for pooled or macro correlation.

    stat_name can be:
        "pooled_spearman_rho"
        "macro_spearman_rho"
        "pooled_pearson_r"
        "macro_pearson_r"
        "pooled_kendall_tau"
        "macro_kendall_tau"
    """
    rng = np.random.default_rng(seed)

    if len(records) < 3:
        return {
            "point": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot_valid": 0,
        }

    def compute_stat(sample_records):
        res = evaluate_association(sample_records)

        if stat_name.startswith("pooled_"):
            key = stat_name.replace("pooled_", "")
            return res["pooled"][key]

        if stat_name.startswith("macro_"):
            return res["macro"][stat_name]

        raise ValueError(f"Unknown stat_name: {stat_name}")

    boot_values = []

    if stratify_by_part:

        parts = sorted(
            set(
                r["part"]
                for r in records
            ),
            key=lambda x: str(x),
        )

        by_part = {
            part: [
                r
                for r in records
                if r["part"] == part
            ]
            for part in parts
        }

        for _ in range(n_boot):

            sample = []

            for part, group in by_part.items():

                n = len(group)

                idx = rng.integers(
                    0,
                    n,
                    size=n,
                )

                sample.extend(
                    [
                        group[i]
                        for i in idx
                    ]
                )

            stat = compute_stat(sample)

            if np.isfinite(stat):
                boot_values.append(stat)

    else:

        n = len(records)

        for _ in range(n_boot):

            idx = rng.integers(
                0,
                n,
                size=n,
            )

            sample = [
                records[i]
                for i in idx
            ]

            stat = compute_stat(sample)

            if np.isfinite(stat):
                boot_values.append(stat)

    boot_values = np.asarray(boot_values, dtype=float)

    point = compute_stat(records)

    if len(boot_values) == 0:
        return {
            "point": float(point) if np.isfinite(point) else np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot_valid": 0,
        }

    lo, hi = np.percentile(
        boot_values,
        [2.5, 97.5],
    )

    return {
        "point": float(point) if np.isfinite(point) else np.nan,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot_valid": int(len(boot_values)),
    }



def permutation_test_within_part(
    records,
    stat_name="macro_spearman_rho",
    n_perm=5000,
    seed=0,
):
    """
    Shuffle distance/error values within each part and recompute statistic.

    For distance metrics, observed statistic is expected to be negative.
    """
    rng = np.random.default_rng(seed)

    if len(records) < 3:
        return {
            "observed": np.nan,
            "p_value": np.nan,
            "n_perm_valid": 0,
        }

    def compute_stat(sample_records):
        res = evaluate_association(sample_records)

        if stat_name.startswith("pooled_"):
            key = stat_name.replace("pooled_", "")
            return res["pooled"][key]

        if stat_name.startswith("macro_"):
            return res["macro"][stat_name]

        raise ValueError(f"Unknown stat_name: {stat_name}")

    observed = compute_stat(records)

    if not np.isfinite(observed):
        return {
            "observed": np.nan,
            "p_value": np.nan,
            "n_perm_valid": 0,
        }

    parts = sorted(
        set(
            r["part"]
            for r in records
        ),
        key=lambda x: str(x),
    )

    indices_by_part = {
        part: [
            i
            for i, r in enumerate(records)
            if r["part"] == part
        ]
        for part in parts
    }

    null_stats = []

    for _ in range(n_perm):

        permuted = [
            dict(r)
            for r in records
        ]

        for part, indices in indices_by_part.items():

            values = np.asarray(
                [
                    records[i]["value"]
                    for i in indices
                ],
                dtype=float,
            )

            values_perm = rng.permutation(values)

            for idx, val in zip(indices, values_perm):
                permuted[idx]["value"] = float(val)

        stat = compute_stat(permuted)

        if np.isfinite(stat):
            null_stats.append(stat)

    null_stats = np.asarray(null_stats, dtype=float)

    if len(null_stats) == 0:
        return {
            "observed": float(observed),
            "p_value": np.nan,
            "n_perm_valid": 0,
        }

    p_value = (
        1.0
        + np.sum(np.abs(null_stats) >= abs(observed))
    ) / (
        1.0
        + len(null_stats)
    )

    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "n_perm_valid": int(len(null_stats)),
        "null_mean": float(np.mean(null_stats)),
        "null_std": float(np.std(null_stats)),
    }


def _append_numeric_control(
    columns,
    values,
    transform=None,
    rank_transform=True,
):
    values = np.asarray(values, dtype=float)

    if transform == "log1p":
        values = np.where(
            values >= 0,
            np.log1p(values),
            np.nan,
        )

    finite = np.isfinite(values)

    if not np.any(finite):
        return False

    values = values.copy()

    if rank_transform:
        values_filled = values.copy()
        mean_value = float(np.nanmean(values_filled))
        values_filled = np.where(finite, values_filled, mean_value)

        values_ranked = rankdata(values_filled)
        values = values_ranked
    else:
        mean_value = float(np.nanmean(values))
        values = np.where(finite, values, mean_value)

    std_value = float(np.std(values))

    if not np.isfinite(std_value) or std_value <= 0:
        return False

    values = (values - float(np.mean(values))) / std_value

    columns.append(values)
    return True


def build_control_matrix(
    records,
    controls=("part", "log_n_common", "silence"),
    rank_numeric_controls=True,
):
    """
    Build control matrix for partial Spearman.

    Supported controls:
        "part"
        "n_common"
        "log_n_common"
        "silence"
    """
    columns = []
    used_controls = []

    columns.append(
        np.ones(
            len(records),
            dtype=float,
        )
    )
    used_controls.append("intercept")

    if "part" in controls:
        parts = sorted(
            set(r["part"] for r in records),
            key=lambda x: str(x),
        )

        for part in parts[1:]:
            col = np.asarray(
                [
                    1.0 if r["part"] == part else 0.0
                    for r in records
                ],
                dtype=float,
            )

            if np.std(col) > 0:
                columns.append(col)
                used_controls.append(f"part={part}")

    if "n_common" in controls:
        ok = _append_numeric_control(
            columns,
            [
                r.get("n_common", np.nan)
                for r in records
            ],
            transform=None,
            rank_transform=rank_numeric_controls,
        )

        if ok:
            used_controls.append("n_common")

    if "log_n_common" in controls:
        ok = _append_numeric_control(
            columns,
            [
                r.get("n_common", np.nan)
                for r in records
            ],
            transform="log1p",
            rank_transform=rank_numeric_controls,
        )

        if ok:
            used_controls.append("log_n_common")

    if "silence" in controls:
        ok = _append_numeric_control(
            columns,
            [
                r.get("silence", np.nan)
                for r in records
            ],
            transform=None,
            rank_transform=rank_numeric_controls,
        )

        if ok:
            used_controls.append("silence")

    X = np.column_stack(columns)

    return X, used_controls


def residualize(y, X):
    """
    Residualize y with respect to control matrix X.
    """
    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    return y - X @ beta


def partial_spearman(records, controls=("part", "n_common")):
    clean_records = [
        r
        for r in records
        if np.isfinite(r["score"])
        and np.isfinite(r["value"])
    ]

    if len(clean_records) < 3:
        return {
            "n": len(clean_records),
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": (),
        }

    y = np.asarray(
        [
            r["score"]
            for r in clean_records
        ],
        dtype=float,
    )

    x = np.asarray(
        [
            r["value"]
            for r in clean_records
        ],
        dtype=float,
    )

    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {
            "n": len(clean_records),
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": (),
        }

    y_rank = rankdata(y)
    x_rank = rankdata(x)

    X_control, used_controls = build_control_matrix(
        clean_records,
        controls=controls,
    )

    y_resid = residualize(
        y_rank,
        X_control,
    )

    x_resid = residualize(
        x_rank,
        X_control,
    )

    if np.std(y_resid) == 0 or np.std(x_resid) == 0:
        return {
            "n": len(clean_records),
            "partial_spearman_rho": np.nan,
            "p_value": np.nan,
            "controls": tuple(controls),
            "used_controls": tuple(used_controls),
        }

    rho, p_value = pearsonr(
        y_resid,
        x_resid,
    )

    return {
        "n": int(len(clean_records)),
        "partial_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "controls": tuple(controls),
        "used_controls": tuple(used_controls),
    }

def evaluate_distance_association_configV1(
    distances,
    encoder,
    dataset,
    layer,
    strategy,
    phone_class,
    alignment,
    rank,
    metric,
    aggregation="mean",
    compute_partial=True,
    bootstrap=True,
    permutation=True,
    n_boot=5000,
    n_perm=5000,
    seed=0,
    return_records=False,
    silence_by_recording: dict = None,
    silence_key: str = "silence_ratio",
):
    """
    Full non-ASA association evaluation for one configuration and one metric.

    Important:
        Uses raw distance.
        Negative correlations are expected.

    Returns:
        {
            "config": ...,
            "pooled": ...,
            "by_part": ...,
            "macro": ...,
            "partial_part": ...,
            "partial_part_coverage": ...,
            optionally "bootstrap_macro_spearman": ...,
            optionally "permutation_macro_spearman": ...,
            optionally "records": ...
        }
    """
    records = collect_records_from_nested_distances(
        distances=distances,
        encoder=encoder,
        dataset=dataset,
        layer=layer,
        strategy=strategy,
        phone_class=phone_class,
        alignment=alignment,
        rank=rank,
        metric=metric,
        aggregation=aggregation,
    )

    if silence_by_recording is not None:
        records = add_silence_to_records(
            records=records,
            silence_by_recording=silence_by_recording,
            silence_key="silence",
            value_key=silence_key,
        )

    eval_result = evaluate_association(records)

    output = {
        "config": {
            "encoder": encoder,
            "dataset": dataset,
            "layer": layer,
            "strategy": strategy,
            "phone_class": phone_class,
            "alignment": alignment,
            "rank": rank,
            "metric": metric,
            "aggregation": aggregation,
        },
        "pooled": eval_result["pooled"],
        "by_part": eval_result["by_part"],
        "macro": eval_result["macro"],
    }

    if compute_partial:
        partial_part = partial_spearman(
            records,
            controls=("part",),
        )

        partial_part_coverage = partial_spearman(
            records,
            controls=("part", "n_common"),
        )

        output["partial_part"] = partial_part
        output["partial_part_coverage"] = partial_part_coverage

        if silence_by_recording is not None:
            partial_part_coverage_silence = partial_spearman(
                records,
                controls=("part", "n_common", "silence"),
            )

            partial_part_logcoverage_silence = partial_spearman(
                records,
                controls=("part", "log_n_common", "silence"),
            )

            output["partial_part_coverage_silence"] = partial_part_coverage_silence
            output["partial_part_logcoverage_silence"] = partial_part_logcoverage_silence

    if bootstrap:
        output["bootstrap_macro_spearman"] = bootstrap_ci(
            records,
            stat_name="macro_spearman_rho",
            n_boot=n_boot,
            seed=seed,
            stratify_by_part=True,
        )

    if permutation:
        output["permutation_macro_spearman"] = permutation_test_within_part(
            records,
            stat_name="macro_spearman_rho",
            n_perm=n_perm,
            seed=seed,
        )

    if return_records:
        output["records"] = records

    return output


def evaluate_distance_association_config(
    distances,
    encoder,
    dataset,
    layer,
    strategy,
    phone_class,
    alignment,
    rank,
    metric,
    aggregation="mean",
    compute_partial=True,
    bootstrap=True,
    permutation=True,
    n_boot=5000,
    n_perm=5000,
    seed=0,
    return_records=False,
    silence_by_recording: dict = None,
    silence_key: str = "silence_ratio",
):
    """
    Full non-ASA association evaluation for one configuration and one metric.

    Important:
        Uses raw distance.
        Negative correlations are expected.

    Returns:
        {
            "config": ...,
            "pooled": ...,
            "by_part": ...,
            "macro": ...,
            "partial_by_part_coverage": ...,
            "partial_macro_coverage": ...,
            "partial_by_part_silence": ...,
            "partial_macro_silence": ...,
            "partial_by_part_coverage_silence": ...,
            "partial_macro_coverage_silence": ...,
            optionally "bootstrap_macro_spearman": ...,
            optionally "permutation_macro_spearman": ...,
            optionally "records": ...
        }
    """
    records = collect_records_from_nested_distances(
        distances=distances,
        encoder=encoder,
        dataset=dataset,
        layer=layer,
        strategy=strategy,
        phone_class=phone_class,
        alignment=alignment,
        rank=rank,
        metric=metric,
        aggregation=aggregation,
    )

    if silence_by_recording is not None:
        records = add_silence_to_records(
            records=records,
            silence_by_recording=silence_by_recording,
            silence_key="silence",
            value_key=silence_key,
        )

    eval_result = evaluate_association(records)

    output = {
        "config": {
            "encoder": encoder,
            "dataset": dataset,
            "layer": layer,
            "strategy": strategy,
            "phone_class": phone_class,
            "alignment": alignment,
            "rank": rank,
            "metric": metric,
            "aggregation": aggregation,
        },
        "pooled": eval_result["pooled"],
        "by_part": eval_result["by_part"],
        "macro": eval_result["macro"],
    }

    if compute_partial:
        def summarize_partial_rhos(rhos: list[float]) -> dict[str, Any]:
            arr = np.asarray(rhos, dtype=float)
            arr = arr[np.isfinite(arr)]

            if arr.size == 0:
                return {
                    "n_parts": 0,
                    "macro_partial_spearman_rho": np.nan,
                    "median_partial_spearman_rho": np.nan,
                    "std_partial_spearman_rho": np.nan,
                }

            return {
                "n_parts": int(arr.size),
                "macro_partial_spearman_rho": fisher_average_correlations(arr),
                "median_partial_spearman_rho": float(np.median(arr)),
                "std_partial_spearman_rho": (
                    float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
                ),
            }

        partial_by_part_coverage = {}
        partial_by_part_silence = {}
        partial_by_part_coverage_silence = {}

        partial_rhos_coverage = []
        partial_rhos_silence = []
        partial_rhos_coverage_silence = []

        for part in sorted(eval_result["by_part"].keys(), key=str):
            part_records = [
                r for r in records
                if str(r["part"]) == str(part)
            ]

            # Partial within this part, controlling for coverage only.
            partial_cov = partial_spearman(
                part_records,
                controls=("n_common",),
            )
            partial_by_part_coverage[str(part)] = partial_cov
            if np.isfinite(partial_cov["partial_spearman_rho"]):
                partial_rhos_coverage.append(partial_cov["partial_spearman_rho"])

            # Partial within this part, controlling for silence only.
            partial_sil = partial_spearman(
                part_records,
                controls=("silence",),
            )
            partial_by_part_silence[str(part)] = partial_sil
            if np.isfinite(partial_sil["partial_spearman_rho"]):
                partial_rhos_silence.append(partial_sil["partial_spearman_rho"])

            # Partial within this part, controlling for coverage and silence.
            partial_cov_sil = partial_spearman(
                part_records,
                controls=("n_common", "silence"),
            )
            partial_by_part_coverage_silence[str(part)] = partial_cov_sil
            if np.isfinite(partial_cov_sil["partial_spearman_rho"]):
                partial_rhos_coverage_silence.append(
                    partial_cov_sil["partial_spearman_rho"]
                )

        output["partial_by_part_coverage"] = partial_by_part_coverage
        output["partial_macro_coverage"] = summarize_partial_rhos(
            partial_rhos_coverage
        )

        output["partial_by_part_silence"] = partial_by_part_silence
        output["partial_macro_silence"] = summarize_partial_rhos(
            partial_rhos_silence
        )

        output["partial_by_part_coverage_silence"] = partial_by_part_coverage_silence
        output["partial_macro_coverage_silence"] = summarize_partial_rhos(
            partial_rhos_coverage_silence
        )

    if bootstrap:
        output["bootstrap_macro_spearman"] = bootstrap_ci(
            records,
            stat_name="macro_spearman_rho",
            n_boot=n_boot,
            seed=seed,
            stratify_by_part=True,
        )

    if permutation:
        output["permutation_macro_spearman"] = permutation_test_within_part(
            records,
            stat_name="macro_spearman_rho",
            n_perm=n_perm,
            seed=seed,
        )

    if return_records:
        output["records"] = records

    output["diagnostics"] = build_record_diagnostics(
        records=records,
        recording_id_key="recording_id",
        n_common_key="n_common",
        silence_key="silence",
    )

    return output




################ Usage and Utils #################################
def aggregate_records_to_recording(
    records,
    weight_col="n_common",
    distance_col="distance",
    silence_col="silence",
):
    """
    Aggregate part-level records into one recording-level record.

    By default:
        - distance: weighted mean using n_common
        - n_common: sum
        - silence: weighted mean using n_common
        - score: first available value
        - level: first available value
        - part: concatenated unique parts
    """
    df = pd.DataFrame(records).copy()

    if len(df) == 0:
        return []

    out = []

    for recording_id, g in df.groupby("recording_id", sort=False):
        item = {
            "recording_id": recording_id,
            "score": g["score"].iloc[0] if "score" in g.columns else np.nan,
            "level": g["level"].iloc[0] if "level" in g.columns else None,
            "part": ",".join(sorted(map(str, g["part"].unique()))) if "part" in g.columns else None,
        }

        # weights
        if weight_col in g.columns:
            w = pd.to_numeric(g[weight_col], errors="coerce").to_numpy(dtype=float)
        else:
            w = np.ones(len(g), dtype=float)

        w = np.where(np.isfinite(w), w, 0.0)

        def weighted_mean(col):
            if col not in g.columns:
                return np.nan

            x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & (w > 0)

            if not np.any(mask):
                return np.nan

            return float(np.average(x[mask], weights=w[mask]))

        # pooled distance
        item["distance"] = weighted_mean(distance_col)

        # pooled silence
        if silence_col in g.columns:
            item["silence"] = weighted_mean(silence_col)
        else:
            item["silence"] = np.nan

        # pooled n_common
        if "n_common" in g.columns:
            item["n_common"] = float(
                np.nansum(pd.to_numeric(g["n_common"], errors="coerce").to_numpy(dtype=float))
            )
        else:
            item["n_common"] = np.nan

        out.append(item)

    return out


def analyze_all_distance_associations(
    distances,
    encoder,
    dataset,
    layers,
    strategies,
    phone_classes,
    alignments,
    ranks,
    metrics,
    aggregation="mean",
    compute_partial=True,
    bootstrap=False,
    permutation=False,
    n_boot=5000,
    n_perm=5000,
    seed=0,
    return_records=False,
    silence_by_recording: dict = None,
    silence_key: str = "silence_ratio",
):
    """
    Run evaluate_distance_association_config() over all requested configurations.

    Uses raw distances.

    Therefore:
        negative correlations are expected.

    Returns:
        results[layer][strategy][phone_class][alignment][rank][metric]
    """
    results = {}

    for layer in layers:

        results[layer] = {}

        for strategy in strategies:

            results[layer][strategy] = {}

            for phone_class in phone_classes:

                # full_unit has no monophone in your setup
                if strategy == "full_unit" and phone_class == "monophone":
                    continue

                results[layer][strategy][phone_class] = {}

                for alignment in alignments:

                    results[layer][strategy][phone_class][alignment] = {}

                    for rank in ranks:

                        results[layer][strategy][phone_class][alignment][rank] = {}

                        for metric in metrics:

                            result = evaluate_distance_association_config(
                                distances=distances,
                                encoder=encoder,
                                dataset=dataset,
                                layer=layer,
                                strategy=strategy,
                                phone_class=phone_class,
                                alignment=alignment,
                                rank=rank,
                                metric=metric,
                                aggregation=aggregation,
                                compute_partial=compute_partial,
                                bootstrap=bootstrap,
                                permutation=permutation,
                                n_boot=n_boot,
                                n_perm=n_perm,
                                seed=seed,
                                return_records=return_records,
                                silence_by_recording=silence_by_recording,
                                silence_key=silence_key,
                            )

                            results[layer][strategy][phone_class][alignment][rank][metric] = result

    return results


def format_float(x, digits=4, signed=True):
    if x is None:
        return "nan"

    if not np.isfinite(x):
        return "nan"

    if signed:
        return f"{x:+.{digits}f}"

    return f"{x:.{digits}f}"


def format_p_value(p):
    if p is None:
        return "nan"

    if not np.isfinite(p):
        return "nan"

    if p < 0.001:
        return f"{p:.2e}"

    return f"{p:.4f}"


def print_association_results(
    results,
    sort_parts=True,
    show_kendall=True,
    show_pearson=True,
    show_summary=True,
):
    """
    Pretty-print association results.

    Compatible with:
        results[layer][strategy][phone_class][alignment][rank][metric] = output

    Important:
        These are raw-distance correlations.
        Negative correlations are expected.
    """

    for layer, strategies in results.items():

        print("\n" + "=" * 100)
        print(f"LAYER {layer}")
        print("=" * 100)

        for strategy, phone_classes in strategies.items():

            print(f"\n  [{strategy}]")

            for phone_class, alignments in phone_classes.items():

                print(f"\n    {phone_class}")

                for alignment, ranks in alignments.items():

                    print(f"      alignment: {alignment}")

                    for rank, metric_outputs in ranks.items():

                        print(f"        rank: {rank}")

                        for metric, output in metric_outputs.items():

                            print(f"          metric: {metric}")

                            by_part = output["by_part"]

                            part_items = list(by_part.items())

                            if sort_parts:
                                try:
                                    part_items = sorted(
                                        part_items,
                                        key=lambda x: int(x[0]),
                                    )
                                except (ValueError, TypeError):
                                    part_items = sorted(
                                        part_items,
                                        key=lambda x: str(x[0]),
                                    )

                            for part, res in part_items:

                                line = (
                                    f"            "
                                    f"part {str(part):<8}"
                                    f"rho = {format_float(res['spearman_rho']):>8}    "
                                    f"p = {format_p_value(res['spearman_p_value']):>10}    "
                                    f"n = {res['n']}"
                                )

                                if show_kendall:
                                    line += (
                                        f"    "
                                        f"tau = {format_float(res['kendall_tau']):>8}"
                                    )

                                if show_pearson:
                                    line += (
                                        f"    "
                                        f"r = {format_float(res['pearson_r']):>8}"
                                    )

                                print(line)

                            if show_summary:

                                pooled = output["pooled"]
                                macro = output["macro"]

                                print(f"            {'-' * 80}")

                                summary_line = (
                                    f"            "
                                    f"SUMMARY     "
                                    f"pooled_rho = {format_float(pooled['spearman_rho']):>8}    "
                                    f"macro_rho = {format_float(macro['macro_spearman_rho']):>8}    "
                                    f"n_total = {macro['n_total']}"
                                )

                                if "partial_part" in output:
                                    summary_line += (
                                        f"    "
                                        f"partial_part = "
                                        f"{format_float(output['partial_part']['partial_spearman_rho']):>8}"
                                    )

                                if "partial_part_coverage" in output:
                                    summary_line += (
                                        f"    "
                                        f"partial_part_cov = "
                                        f"{format_float(output['partial_part_coverage']['partial_spearman_rho']):>8}"
                                    )

                                print(summary_line)

                                if "bootstrap_macro_spearman" in output:

                                    boot = output["bootstrap_macro_spearman"]

                                    print(
                                        f"            "
                                        f"BOOTSTRAP   "
                                        f"macro_rho = {format_float(boot['point']):>8}    "
                                        f"95% CI = ["
                                        f"{format_float(boot['ci_lower'])}, "
                                        f"{format_float(boot['ci_upper'])}]    "
                                        f"valid = {boot['n_boot_valid']}"
                                    )

                                if "permutation_macro_spearman" in output:

                                    perm = output["permutation_macro_spearman"]

                                    print(
                                        f"            "
                                        f"PERMUTATION "
                                        f"observed = {format_float(perm['observed']):>8}    "
                                        f"p = {format_p_value(perm['p_value'])}    "
                                        f"valid = {perm['n_perm_valid']}"
                                    )



def coverage_correlation(records):
    """
    Correlate CEFR score with number of common phones/diphones/triphones.

    Positive correlation means higher CEFR recordings have more common units.
    """
    clean = [
        r
        for r in records
        if np.isfinite(r["score"])
        and np.isfinite(r.get("n_common", np.nan))
    ]

    scores = [
        r["score"]
        for r in clean
    ]

    coverage = [
        r["n_common"]
        for r in clean
    ]

    spearman = safe_corr(
        scores,
        coverage,
        method="spearman",
    )

    kendall = safe_corr(
        scores,
        coverage,
        method="kendall",
    )

    pearson = safe_corr(
        scores,
        coverage,
        method="pearson",
    )

    return {
        "n": spearman["n"],

        "coverage_spearman_rho": spearman["stat"],
        "coverage_spearman_p_value": spearman["p_value"],

        "coverage_kendall_tau": kendall["stat"],
        "coverage_kendall_p_value": kendall["p_value"],

        "coverage_pearson_r": pearson["stat"],
        "coverage_pearson_p_value": pearson["p_value"],
    }







def _parse_seed_name(
    name: str,
    seed_prefix: str = "seed",
) -> int | str | None:
    """
    Parse folder name like 'seed0' -> 0.
    """
    if not name.startswith(seed_prefix):
        return None

    suffix = name[len(seed_prefix):]

    if suffix == "":
        return name

    try:
        return int(suffix)
    except ValueError:
        return name




def summarize_association_grid(results):
    """
    Flatten results from analyze_all_distance_associations into a list.

    Since values are raw distances, more negative macro_spearman_rho is better.
    """
    rows = []

    for layer, strategies in results.items():

        for strategy, phone_classes in strategies.items():

            for phone_class, alignments in phone_classes.items():

                for alignment, ranks in alignments.items():

                    for rank, metric_outputs in ranks.items():

                        for metric, output in metric_outputs.items():

                            pooled = output["pooled"]
                            macro = output["macro"]

                            row = {
                                "layer": layer,
                                "strategy": strategy,
                                "phone_class": phone_class,
                                "alignment": alignment,
                                "rank": rank,
                                "metric": metric,

                                "pooled_spearman_rho": pooled["spearman_rho"],
                                "pooled_spearman_p_value": pooled["spearman_p_value"],

                                "macro_spearman_rho": macro["macro_spearman_rho"],
                                "macro_kendall_tau": macro["macro_kendall_tau"],
                                "macro_pearson_r": macro["macro_pearson_r"],

                                "n_total": macro["n_total"],
                                "n_parts": macro["n_parts"],
                            }

                            if "partial_part" in output:
                                row["partial_spearman_part"] = output[
                                    "partial_part"
                                ]["partial_spearman_rho"]

                            if "partial_part_coverage" in output:
                                row["partial_spearman_part_coverage"] = output[
                                    "partial_part_coverage"
                                ]["partial_spearman_rho"]

                            # Add per-part Spearman values.
                            for part, part_res in output["by_part"].items():
                                row[f"rho_part_{part}"] = part_res["spearman_rho"]
                                row[f"n_part_{part}"] = part_res["n"]

                            rows.append(row)

    # Sort by macro Spearman ascending because more negative is better.
    rows = sorted(
        rows,
        key=lambda r: (
            np.inf
            if not np.isfinite(r["macro_spearman_rho"])
            else r["macro_spearman_rho"]
        ),
    )

    return rows

def build_record_diagnostics(
    records,
    recording_id_key: str = "recording_id",
    n_common_key: str = "n_common",
    silence_key: str = "silence",
) -> dict[str, Any]:
    """
    Build diagnostics similar to the UME-ERJ diagnostics block.

    Works for both SANDI part-level records and UME-ERJ recording-level records.
    """
    df = pd.DataFrame(records).copy()

    diagnostics: dict[str, Any] = {
        "n_records": int(len(df)),
        "n_unique_recordings": (
            int(df[recording_id_key].nunique())
            if recording_id_key in df.columns
            else 0
        ),
    }

    for col in [n_common_key, silence_key]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(x)

            diagnostics[f"finite_{col}"] = int(finite.sum())
            diagnostics[f"unique_{col}"] = (
                int(len(np.unique(x[finite])))
                if np.any(finite)
                else 0
            )

            if np.any(finite):
                diagnostics[f"min_{col}"] = float(np.min(x[finite]))
                diagnostics[f"mean_{col}"] = float(np.mean(x[finite]))
                diagnostics[f"max_{col}"] = float(np.max(x[finite]))
        else:
            diagnostics[f"finite_{col}"] = 0
            diagnostics[f"unique_{col}"] = 0

    return diagnostics

#################UME_ERJ###############################
###############################################################################
# UME-ERJ: distance evaluation by individual rater
###############################################################################
# =====================================================================
# Expected external constant from your project
# =====================================================================
def _load_pickle_or_empty(path: Path | None) -> Any:
    """
    Load a pickle if it exists, otherwise return an empty dict.
    Also handles None safely.
    """
    if path is None:
        return {}

    path = Path(path)

    if not path.exists():
        return {}

    obj = load_pickle(path)

    if obj is None:
        return {}

    return obj


def _parse_seed_folder_name(
    name: str,
    seed_prefix: str = "seed",
) -> int | str | None:
    """
    Parse folder names like:
        seed0 -> 0
        random_seed0 -> 0

    Returns an int if the suffix is numeric, otherwise the original name.
    """
    prefixes = [seed_prefix]

    if seed_prefix != "seed":
        prefixes.append("seed")

    if seed_prefix != "random_seed":
        prefixes.append("random_seed")

    for prefix in prefixes:
        if name.startswith(prefix):
            suffix = name[len(prefix):]

            if suffix == "":
                return name

            try:
                return int(suffix)
            except ValueError:
                return name

    return None


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_distance_pickles(
    models,
    distance_root,
    datasets=("train", "dev"),
    per_sample=True,
    projection_mode: str = "svd",
    random_seeds: list[int] | None = None,
    seed_prefix: str = "seed",
    params: dict[str, Any] | None = None,
):
    """
    Load previously computed SANDI distance pickles.

    Notes
    -----
    - `distance_root` must already point to the projection folder that
      directly contains encoder folders.
    - For noisy cosine experiments, `params` must match the parameters
      used when saving.
    - For noisy whitened runs, use:
        projection_mode="whitened"
      not "svd".
    """
    distance_root = Path(distance_root)
    distances = {}

    if projection_mode not in {"svd", "random", "whitened", "identity"}:
        raise ValueError(
            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
        )

    noise_suffix = make_cosine_noise_suffix(params)

    for encoder_name, info in models.items():
        distances[encoder_name] = {}

        for layer in info["layer"]:
            layer_dir = distance_root / encoder_name / f"layer{layer}"

            if projection_mode == "random":
                if random_seeds is None:
                    discovered_seeds = []

                    if layer_dir.exists():
                        for child in layer_dir.iterdir():
                            if child.is_dir():
                                parsed = _parse_seed_folder_name(
                                    child.name,
                                    seed_prefix=seed_prefix,
                                )
                                if parsed is not None:
                                    discovered_seeds.append(parsed)

                    seed_values = sorted(set(discovered_seeds), key=str)
                else:
                    seed_values = list(random_seeds)

                distances[encoder_name][layer] = {}

                for seed in seed_values:
                    distances[encoder_name][layer][seed] = {}

                    seed_dirs = [
                        layer_dir / f"{seed_prefix}{seed}",
                        layer_dir / f"seed{seed}",
                        layer_dir / f"random_seed{seed}",
                    ]

                    for dataset in datasets:
                        filename = (
                            f"{dataset}"
                            f"{'_sample' if per_sample else ''}"
                            f"{noise_suffix}.pickle"
                        )

                        candidates = [
                            d / filename
                            for d in seed_dirs
                        ]

                        path = _first_existing_path(candidates)

                        if path is None:
                            print(
                                f"WARNING: missing file for "
                                f"encoder={encoder_name}, layer={layer}, "
                                f"seed={seed}, dataset={dataset}"
                            )
                            print("Tried:")
                            for c in candidates:
                                print("  ", c)

                            distances[encoder_name][layer][seed][dataset] = {}
                            continue

                        print(f"Loading {path}")
                        distances[encoder_name][layer][seed][dataset] = (
                            _load_pickle_or_empty(path)
                        )

            else:
                distances[encoder_name][layer] = {}

                for dataset in datasets:
                    filename = (
                        f"{dataset}"
                        f"{'_sample' if per_sample else ''}"
                        f"{noise_suffix}.pickle"
                    )

                    path = layer_dir / filename

                    if not path.exists():
                        print(
                            f"WARNING: missing file for "
                            f"encoder={encoder_name}, layer={layer}, dataset={dataset}"
                        )
                        print("Tried:")
                        print("  ", path)
                    else:
                        print(f"Loading {path}")

                    distances[encoder_name][layer][dataset] = (
                        _load_pickle_or_empty(path)
                    )

    return distances




