from typing import Any, Iterable, Optional

import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from config import UMEERJ_INVALID_SCORES

from build_phone_codes import (
    AveragingStrategy,
)

from utils import (
    get_transformed,
    load_pickle,
    make_cosine_noise_suffix,
    PhoneClass,
    save_pickle,
    stable_seed,

)

from projections import (
    make_random_projection_codebook,
    make_power_whitened_codebook,
    apply_white_noise_for_cosine,
)





def make_identity_codebook(
    base_codebook: dict[str, Any],
    native_matrix_key: str = "X_ref_all",
    native_phones_key: str = "phones",
) -> dict[str, Any]:
    """
    Build a codebook that keeps the native averages as-is:
    - U = native matrix
    - sigma = ones
    - VT = identity
    """
    X_native = np.asarray(base_codebook[native_matrix_key], dtype=np.float64)

    if X_native.ndim != 2:
        raise ValueError(
            f"{native_matrix_key} must be 2D, got shape {X_native.shape}."
        )

    n_features = X_native.shape[1]

    return {
        **base_codebook,
        "U": X_native,
        "sigma": np.ones(n_features, dtype=np.float64),
        "VT": np.eye(n_features, dtype=np.float64),
        "phones": base_codebook[native_phones_key],
        "global_mean": np.zeros(n_features, dtype=np.float64),
    }


def attach_silence_to_records(
    records: list[dict[str, Any]],
    silence_by_recording: dict[Any, float],
    silence_key: str = "silence",
) -> list[dict[str, Any]]:
    """
    Attach silence values to record dictionaries.

    silence_by_recording can use either:
        recording_id
    or:
        (part, recording_id)

    as key.
    """
    output = []

    for r in records:
        item = dict(r)

        recording_id = str(item["recording_id"])
        part = str(item.get("part", ""))

        value = np.nan

        if (part, recording_id) in silence_by_recording:
            value = silence_by_recording[(part, recording_id)]
        elif recording_id in silence_by_recording:
            value = silence_by_recording[recording_id]

        try:
            value = float(value)
        except Exception:
            value = np.nan

        item[silence_key] = value
        output.append(item)

    return output

def create_comparable_matrices(
    level_pairs: dict[str, dict[Any, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Create comparable reference/test matrices for each level.

    Parameters
    ----------
    level_pairs
        Mapping from level to phone/n_phone pairs. Each pair must contain
        ``"ref"`` and ``"test"`` arrays.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Mapping from level to dictionaries containing ``"ref"`` and
        ``"test"`` matrices.
    """
    matrices: dict[str, dict[str, np.ndarray]] = {}

    for level, phone_data in level_pairs.items():
        phones = sorted(phone_data)

        if not phones:
            print(f"WARNING: no common n-phones for {level}")
            continue

        matrices[level] = {
            "ref": np.stack([phone_data[phone]["ref"] for phone in phones]),
            "test": np.stack([phone_data[phone]["test"] for phone in phones]),
        }

    return matrices


def get_pairs_for_test_phones(
    X_test: np.ndarray,
    phone_dict_test: dict[Any, int],
    codebook: dict[str, Any],
    global_mean_normalization: bool = True,
) -> dict[Any, dict[str, np.ndarray]]:
    """
    Construct matched reference/test representation pairs.

    Test phones are matched against the phone/n_phone labels in the
    reference codebook. Matching is case-insensitive.

    Parameters
    ----------
    X_test
        Test representation matrix of shape ``(n_phones, n_features)``.

    phone_dict_test
        Mapping from test phone/n_phone labels to row indices in ``X_test``.

    codebook
        Reference codebook containing ``U``, ``sigma``, ``VT``, ``phones``,
        and optionally ``global_mean``.

    global_mean_normalization
        If True, subtract ``codebook["global_mean"]`` from the test
        representation before projection.

    Returns
    -------
    dict
        Mapping from matched phone/n_phone labels to reference and test
        representation vectors.
    """
    vt_ref = codebook["VT"]
    c_ref = codebook["U"] * codebook["sigma"]

    codebook_phone_dict = {
        tuple(part.upper() for part in n_phone): i
        for i, n_phone in enumerate(codebook["phones"])
    }

    X_test = np.asarray(
        X_test,
        dtype=np.float64,
    )

    if global_mean_normalization:
        global_mean = codebook["global_mean"]

        if global_mean is None:
            raise ValueError(
                "global_mean_normalization=True, but "
                "codebook['global_mean'] is None."
            )

        X_test = X_test - global_mean

    transformed_test = get_transformed(
        X_test,
        vt_ref,
    )

    pairs: dict[Any, dict[str, np.ndarray]] = {}

    for phone, test_index in phone_dict_test.items():
        normalized_phone = tuple(part.upper() for part in phone)

        if normalized_phone not in codebook_phone_dict:
            continue

        ref_index = codebook_phone_dict[normalized_phone]

        pairs[phone] = {
            "ref": c_ref[ref_index],
            "test": transformed_test[test_index],
        }

    return pairs


def get_pairs_for_level_cuts_(
    cuts_by_score: dict[float, Any],
    index: int,
    codebook: dict[str, Any],
    test_averaged_phones: dict[int, list[Any]],
    levels: dict[str, float] | None = None,
    global_mean_normalization: bool = True,
) -> dict[str, dict[Any, dict[str, np.ndarray]]]:
    """
    Construct matched reference/test pairs for one sample at each level.

    Parameters
    ----------
    cuts_by_score
        Mapping from proficiency scores to available cuts.

    index
        Index of the sample to select within each proficiency level.

    codebook
        Reference codebook.

    test_averaged_phones
        Mapping from cut identifiers to averaged test representations and
        their phone/n_phone labels.

    levels
        Mapping from level names to proficiency scores. If None, the
        standard A2-C1 levels are used.

    global_mean_normalization
        Whether to apply codebook global mean normalization.

    Returns
    -------
    dict
        Mapping from proficiency level to matched reference/test pairs.
    """
    if levels is None:
        levels = {
            "a2": 2.0,
            "a2+": 2.5,
            "b1": 3.0,
            "b1+": 3.5,
            "b2": 4.0,
            "b2+": 4.5,
            "c1": 5.0,
        }

    level_pairs: dict[
        str,
        dict[Any, dict[str, np.ndarray]],
    ] = {}

    for level, score in levels.items():
        cuts = cuts_by_score.get(score)

        if cuts is None or index >= len(cuts):
            continue

        cut = cuts[index]

        key = cut[1].recording_id
        X_test = test_averaged_phones[key][0]
        phones_test = test_averaged_phones[key][1]

        phone_dict_test = {phone: i for i, phone in enumerate(phones_test)}

        level_pairs[level] = get_pairs_for_test_phones(
            X_test=X_test,
            phone_dict_test=phone_dict_test,
            codebook=codebook,
            global_mean_normalization=(global_mean_normalization),
        )

    return level_pairs




class StableMatchedMahalanobis:
    """
    Stable Mahalanobis distance for row-matched matrices.

    Two covariance models are supported:

    ``diag``
        Uses only feature-wise variances.

    ``full``
        Uses the full regularized covariance matrix.
    """

    def __init__(
        self,
        mode: str = "diag",
        alpha: float = 0.1,
        var_floor: float = 1e-3,
        gamma: float = 0.5,
        normalize_by_dim: bool = True,
        eps: float = 1e-12,
    ) -> None:
        """
        Initialize the Mahalanobis estimator.

        Parameters
        ----------
        mode
            ``"diag"`` or ``"full"``.

        alpha
            Shrinkage toward isotropic variance.

        var_floor
            Relative floor for variances/eigenvalues.

        gamma
            Exponent applied to diagonal variances.

        normalize_by_dim
            If True, divide squared distances by representation
            dimension.

        eps
            Numerical stability constant.
        """
        if mode not in {"diag", "full"}:
            raise ValueError("mode must be 'diag' or 'full'.")

        self.mode = mode
        self.alpha = float(alpha)
        self.var_floor = float(var_floor)
        self.gamma = float(gamma)
        self.normalize_by_dim = bool(normalize_by_dim)
        self.eps = float(eps)

    def fit(
        self,
        Z: np.ndarray,
    ) -> "StableMatchedMahalanobis":
        """
        Fit covariance/variance parameters from reference samples.

        Parameters
        ----------
        Z
            Matrix of shape ``(N, K)``.

        Returns
        -------
        StableMatchedMahalanobis
            Fitted estimator.
        """
        Z = np.asarray(
            Z,
            dtype=float,
        )

        if Z.ndim != 2:
            raise ValueError("Z must have shape [N, K].")

        finite = np.isfinite(Z).all(axis=1)
        Z = Z[finite]

        if Z.shape[0] < 2:
            raise ValueError("Need at least two finite rows to fit covariance.")

        self.K_ = Z.shape[1]

        if self.mode == "diag":
            variance = np.var(
                Z,
                axis=0,
                ddof=1,
            )

            self._fit_from_variance(variance)

        else:
            Z_centered = Z - np.mean(
                Z,
                axis=0,
                keepdims=True,
            )

            df = max(
                Z_centered.shape[0] - 1,
                1,
            )

            covariance = Z_centered.T @ Z_centered / df

            covariance = (covariance + covariance.T) / 2.0

            trace = float(np.trace(covariance))

            tau = trace / self.K_ if trace > 0 and np.isfinite(trace) else 1.0

            identity = np.eye(self.K_)

            covariance_reg = (
                1.0 - self.alpha
            ) * covariance + self.alpha * tau * identity

            eigvals, eigvecs = np.linalg.eigh(covariance_reg)

            eig_floor = self.var_floor * tau

            eigvals = np.maximum(
                eigvals,
                eig_floor,
            )

            covariance_reg = (eigvecs * eigvals) @ eigvecs.T

            covariance_reg = (covariance_reg + covariance_reg.T) / 2.0

            self.Sigma_reg_ = covariance_reg

            self.L_ = np.linalg.cholesky(covariance_reg)

        return self

    def _fit_from_variance(
        self,
        var: np.ndarray,
    ) -> None:
        """
        Fit the diagonal covariance model from feature variances.
        """
        variance = np.asarray(
            var,
            dtype=float,
        ).ravel()

        variance = np.nan_to_num(
            variance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        positive = variance[variance > 0]

        tau = float(np.mean(positive)) if len(positive) > 0 else 1.0

        tau = max(
            tau,
            self.eps,
        )

        variance_reg = (1.0 - self.alpha) * variance + self.alpha * tau

        variance_reg = np.maximum(
            variance_reg,
            self.var_floor * tau,
        )

        self.var_reg_ = variance_reg

        self.denom_ = variance_reg**self.gamma

    def distances(
        self,
        C1: np.ndarray,
        C2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute row-wise Mahalanobis distances.

        Parameters
        ----------
        C1
            First matched representation matrix.

        C2
            Second matched representation matrix.

        Returns
        -------
        np.ndarray
            Distance vector of shape ``(m,)``.
        """
        C1 = np.asarray(
            C1,
            dtype=float,
        )

        C2 = np.asarray(
            C2,
            dtype=float,
        )

        if C1.shape != C2.shape:
            raise ValueError(
                "C1 and C2 must have same shape. " f"Got {C1.shape} and {C2.shape}."
            )

        if C1.ndim != 2:
            raise ValueError("C1 and C2 must be 2D matrices.")

        if C1.shape[1] != self.K_:
            raise ValueError(f"Expected K={self.K_}, " f"got K={C1.shape[1]}.")

        _, K = C1.shape

        output = np.full(
            C1.shape[0],
            np.nan,
            dtype=float,
        )

        finite = np.isfinite(C1).all(axis=1) & np.isfinite(C2).all(axis=1)

        if not np.any(finite):
            return output

        delta = C2[finite] - C1[finite]

        if self.mode == "diag":
            squared_distance = np.sum(
                (delta**2) / self.denom_[None, :],
                axis=1,
            )

        else:
            transformed = np.linalg.solve(
                self.L_,
                delta.T,
            ).T

            squared_distance = np.sum(
                transformed**2,
                axis=1,
            )

        if self.normalize_by_dim:
            squared_distance /= K

        output[finite] = np.sqrt(
            np.maximum(
                squared_distance,
                0.0,
            )
        )

        return output


def rowwise_euclidean_distance(
    C_ref: np.ndarray,
    C_test: np.ndarray,
    normalize_by_dim: bool = True,
) -> np.ndarray:
    """
    Compute row-wise Euclidean distances.
    """
    C_ref = np.asarray(
        C_ref,
        dtype=np.float64,
    )

    C_test = np.asarray(
        C_test,
        dtype=np.float64,
    )

    if C_ref.shape != C_test.shape:
        raise ValueError(
            "C_ref and C_test must have the same shape, "
            f"got {C_ref.shape} and {C_test.shape}."
        )

    if C_ref.ndim != 2:
        raise ValueError("C_ref and C_test must be 2-dimensional.")

    delta = C_test - C_ref

    squared_distance = np.sum(
        delta**2,
        axis=1,
    )

    if normalize_by_dim:
        squared_distance /= C_ref.shape[1]

    return np.sqrt(
        np.maximum(
            squared_distance,
            0.0,
        )
    )


def rowwise_cosine_distance(
    C_ref: np.ndarray,
    C_test: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute row-wise cosine distances.
    """
    C_ref = np.asarray(
        C_ref,
        dtype=np.float64,
    )

    C_test = np.asarray(
        C_test,
        dtype=np.float64,
    )

    if C_ref.shape != C_test.shape:
        raise ValueError(
            "C_ref and C_test must have the same shape, "
            f"got {C_ref.shape} and {C_test.shape}."
        )

    if C_ref.ndim != 2:
        raise ValueError("C_ref and C_test must be 2-dimensional.")

    dot_product = np.sum(
        C_ref * C_test,
        axis=1,
    )

    norm_ref = np.linalg.norm(
        C_ref,
        axis=1,
    )

    norm_test = np.linalg.norm(
        C_test,
        axis=1,
    )

    denominator = np.maximum(
        norm_ref * norm_test,
        eps,
    )

    cosine_similarity = dot_product / denominator

    cosine_similarity = np.clip(
        cosine_similarity,
        -1.0,
        1.0,
    )

    return 1.0 - cosine_similarity


def frobenius_distance(
    C_ref: np.ndarray,
    C_test: np.ndarray,
    normalize_by_size: bool = True,
) -> float:
    """
    Compute global Frobenius distance.

    If normalize_by_size=True, return RMS Frobenius distance:
        ||C_ref - C_test||_F / sqrt(m*K)
    """
    C_ref = np.asarray(C_ref, dtype=np.float64)
    C_test = np.asarray(C_test, dtype=np.float64)

    if C_ref.shape != C_test.shape:
        raise ValueError("C_ref and C_test must have the same shape.")

    if C_ref.ndim != 2:
        raise ValueError("C_ref and C_test must be 2D.")

    dist = np.linalg.norm(C_ref - C_test, ord="fro")

    if normalize_by_size:
        dist = dist / np.sqrt(C_ref.size)

    return float(dist)


def procrustes_distance(
    C_ref: np.ndarray,
    C_test: np.ndarray,
    normalize_by_size: bool = True,
) -> float:
    """
    Compute global orthogonal Procrustes distance.
    This is useful to see the distance also in terms of rotation

    If normalize_by_size=True, return RMS residual after alignment:
        ||C_ref - C_test R||_F / sqrt(m*K)
    """
    C_ref = np.asarray(C_ref, dtype=np.float64)
    C_test = np.asarray(C_test, dtype=np.float64)

    if C_ref.shape != C_test.shape:
        raise ValueError("C_ref and C_test must have the same shape.")

    if C_ref.ndim != 2:
        raise ValueError("C_ref and C_test must be 2D.")

    rotation, _ = orthogonal_procrustes(C_test, C_ref)

    C_test_aligned = C_test @ rotation

    dist = np.linalg.norm(C_ref - C_test_aligned, ord="fro")

    if normalize_by_size:
        dist = dist / np.sqrt(C_ref.size)

    return float(dist)


def calculate_maha(
    C_ref: np.ndarray,
    C_test: np.ndarray,
    C_ref_all: np.ndarray,
    mode: str = "diag",
    alpha: float = 0.1,
    var_floor: float = 1e-3,
    gamma: float = 0.5,
    normalize_by_dim: bool = True,
) -> np.ndarray:
    """
    Calculate stable matched Mahalanobis distances.

    Parameters
    ----------
    C_ref
        Reference matrix for the comparison.

    C_test
        Test matrix matched row-by-row with ``C_ref``.

    C_ref_all
        Reference samples used to fit the covariance/variance model.
        If None, ``C_ref`` is used.

    mode
        ``"diag"`` or ``"full"``.

    alpha
        Shrinkage parameter.

    var_floor
        Variance/eigenvalue floor.

    gamma
        Variance exponent.

    normalize_by_dim
        Normalize squared distance by representation dimensionality.

    Returns
    -------
    np.ndarray
        Row-wise Mahalanobis distances.
    """

    maha = StableMatchedMahalanobis(
        mode=mode,
        alpha=alpha,
        var_floor=var_floor,
        gamma=gamma,
        normalize_by_dim=normalize_by_dim,
    )

    maha.fit(C_ref_all)

    return maha.distances(
        C_ref,
        C_test,
    )


def compute_all_representation_distances(
        C_ref: np.ndarray,
        C_test: np.ndarray,
        C_ref_all: np.ndarray,
        maha_mode: str = "diag",
        maha_alpha: float = 0.1,
        maha_var_floor: float = 1e-3,
        maha_gamma: float = 0.5,
        normalize_by_dim: bool = True,
        cosine_noise_mode: str = "none",
        cosine_noise_level: float = 1.0,
        cosine_noise_seed: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Compute the five representation distances.

    The five metrics are:

    - row-wise Euclidean distance
    - global Frobenius distance
    - row-wise cosine distance
    - global orthogonal Procrustes distance
    - row-wise stable Mahalanobis distance

    Parameters
    ----------
    C_ref
        Reference representation matrix.

    C_test
        Test representation matrix matched row-by-row with ``C_ref``.

    C_ref_all
        Global reference samples used to fit the Mahalanobis model.
        If None, ``C_ref`` is used.

    maha_mode
        Stable Mahalanobis mode: ``"diag"`` or ``"full"``.

    maha_alpha
        Mahalanobis shrinkage parameter.

    maha_var_floor
        Mahalanobis variance/eigenvalue floor.

    maha_gamma
        Mahalanobis variance exponent.

    normalize_by_dim
        Whether Euclidean and Mahalanobis distances are normalized by
        representation dimensionality.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the five distance metrics.
    """
    C_ref = np.asarray(
        C_ref,
        dtype=np.float64,
    )

    C_test = np.asarray(
        C_test,
        dtype=np.float64,
    )

    if C_ref.ndim != 2 or C_test.ndim != 2:
        raise ValueError("Both inputs must be 2-dimensional arrays.")

    if C_ref.shape != C_test.shape:
        raise ValueError(
            "Input shapes must match, " f"got {C_ref.shape} and {C_test.shape}."
        )

    if C_ref.shape[0] == 0:
        raise ValueError("Cannot compute distances with zero rows.")

    d_euclidean_per_phone = rowwise_euclidean_distance(
        C_ref,
        C_test,
        normalize_by_dim=normalize_by_dim,
    )

    D_frobenius = frobenius_distance(
        C_ref,
        C_test,
    )

    C_ref_for_cosine, C_test_for_cosine = apply_white_noise_for_cosine(
        C_ref=C_ref,
        C_test=C_test,
        mode=cosine_noise_mode,
        noise_level=cosine_noise_level,
        random_state=cosine_noise_seed,
    )

    d_cosine_per_phone = rowwise_cosine_distance(
        C_ref_for_cosine,
        C_test_for_cosine,
    )

    D_procrustes = procrustes_distance(
        C_ref,
        C_test,
    )

    d_mahalanobis_per_phone = calculate_maha(
        C_ref=C_ref,
        C_test=C_test,
        C_ref_all=C_ref_all,
        mode=maha_mode,
        alpha=maha_alpha,
        var_floor=maha_var_floor,
        gamma=maha_gamma,
        normalize_by_dim=normalize_by_dim,
    )

    return {
        "d_euclidean_per_phone": (d_euclidean_per_phone),
        "D_frobenius": D_frobenius,
        "d_cosine_per_phone": (d_cosine_per_phone),
        "D_procrustes": D_procrustes,
        "d_mahalanobis_per_phone": (d_mahalanobis_per_phone),
    }





def collect_sandi_cuts_by_score(
    cuts: Iterable[Any],
    upper_bound: int = 100,
    score_name: str = "part_score",
    indices: set[int] | None = None,
) -> dict[float, list[tuple[int, Any]]]:
    scores = [x / 2 for x in range(4, 12)]
    result = {score: [] for score in scores}

    for idx, cut in enumerate(cuts):
        if indices is not None and idx not in indices:
            continue
        score = float(cut.supervisions[0].custom[score_name])
        score = int(score * 2) / 2
        if score in result and len(result[score]) < upper_bound:
            result[score].append((idx, cut))
    return result

def compute_overall_representation_distances(
        cuts_by_score: dict[float, Any],
        codebook: dict[str, Any],
        test_averaged_phones: dict[int, Any],
        n_cuts: int = 30,
        levels: dict[str, Any] | None = None,
        pairs: dict[int, dict[str, Any]] | None = None,
        global_mean_normalization: bool = True,
        maha_mode: str = "diag",
        maha_alpha: float = 0.1,
        maha_var_floor: float = 1e-3,
        maha_gamma: float = 0.5,
        normalize_by_dim: bool = True,
        cosine_noise_mode: str = "none",
        cosine_noise_level: float = 1.0,
        cosine_noise_seed: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Compute representation distances across cuts and proficiency levels.

    Returns
    -------
    dict
        Mapping ``level -> metric -> list[distance]``.
    """

    overall_distances: dict[
        str,
        dict[str, list[float]],
    ] = {}

    for i in tqdm(range(n_cuts)):
        if pairs is not None:
            level_pairs = pairs.get(i)
        else:
            level_pairs = get_pairs_for_level_cuts_(
                cuts_by_score=cuts_by_score,
                index=i,
                codebook=codebook,
                test_averaged_phones=(test_averaged_phones),
                levels=levels,
                global_mean_normalization=(global_mean_normalization),
            )

        if not level_pairs:
            continue

        matrices = create_comparable_matrices(level_pairs)

        for level, level_matrices in matrices.items():
            ref_matrix = level_matrices["ref"]
            test_matrix = level_matrices["test"]

            try:
                local_cosine_seed = None
                if cosine_noise_seed is not None:
                    local_cosine_seed = stable_seed(
                        cosine_noise_seed,
                        level,
                        i,
                    )
                distances = compute_all_representation_distances(
                    C_ref=ref_matrix,
                    C_test=test_matrix,
                    C_ref_all=codebook["U"] * codebook["sigma"],
                    maha_mode=maha_mode,
                    maha_alpha=maha_alpha,
                    maha_var_floor=maha_var_floor,
                    maha_gamma=maha_gamma,
                    normalize_by_dim=normalize_by_dim,
                    cosine_noise_mode=cosine_noise_mode,
                    cosine_noise_level=cosine_noise_level,
                    cosine_noise_seed=local_cosine_seed,
                )
            except (
                ValueError,
                np.linalg.LinAlgError,
            ):
                continue

            if level not in overall_distances:
                overall_distances[level] = {metric: [] for metric in distances}

            for metric, value in distances.items():
                value_array = np.asarray(
                    value,
                    dtype=float,
                )

                if not np.all(np.isfinite(value_array)):
                    continue

                aggregated_value = float(np.mean(value_array))

                overall_distances[level][metric].append(aggregated_value)

    return overall_distances





def collect_sandi_cuts_by_part(
    cuts: Iterable[Any],
    upper_bound: int = 100,
    indices: Optional[set[int]] = None,
) -> dict[str, dict[float, list[tuple[int, Any]]]]:
    scores = [x / 2 for x in range(4, 12)]
    result: dict[str, dict[float, list[tuple[int, Any]]]] = {}

    for idx, cut in enumerate(cuts):
        if indices is not None and idx not in indices:
            continue

        supervision = cut.supervisions[0]
        part = str(supervision.custom["part"])
        score = int(float(supervision.custom["part_score"]) * 2) / 2

        if score not in scores:
            continue
        if part not in result:
            result[part] = {s: [] for s in scores}
        if len(result[part][score]) < upper_bound:
            result[part][score].append((idx, cut))

    return result





def compute_overall_representation_distances_by_part(
    cuts_by_part: dict[
        str,
        dict[float, list[tuple[int, Any]]],
    ],
    codebook: dict[str, Any],
    test_averaged_phones: dict[int, Any],
    n_cuts: int = 30,
    levels: dict[str, Any] = None,
    pairs: dict[
        str,
        dict[int, dict[str, Any]],
    ]= None,
    global_mean_normalization: bool = True,
    maha_mode: str = "diag",
    maha_alpha: float = 0.1,
    maha_var_floor: float = 1e-3,
    maha_gamma: float = 0.5,
    normalize_by_dim: bool = True,
    cosine_noise_mode: str = "none",
    cosine_noise_level: float = 1.0,
    cosine_noise_seed: int = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Compute representation distances independently for each part.

    Parameters
    ----------
    cuts_by_part
        Mapping from part name to proficiency-level cuts.

    codebook
        Reference representation codebook.

    test_averaged_phones
        Test phone/n_phone representations.

    n_cuts
        Number of cuts to process per part.

    levels
        Proficiency levels to process.

    pairs
        Optional precomputed pairs indexed by part and cut.


    global_mean_normalization
        Whether to subtract the codebook global mean.

    maha_mode
        Stable Mahalanobis mode: ``"diag"`` or ``"full"``.

    maha_alpha
        Mahalanobis shrinkage parameter.

    maha_var_floor
        Mahalanobis variance/eigenvalue floor.

    maha_gamma
        Mahalanobis variance exponent.

    normalize_by_dim
        Whether Euclidean and Mahalanobis distances are normalized
        by representation dimensionality.

    Returns
    -------
    dict
        Mapping ``part -> level -> metric -> list[distance]``.
    """

    overall_distances: dict[
        str,
        dict[str, dict[str, list[float]]],
    ] = {}

    for part, cuts_by_score in cuts_by_part.items():
        part_distances: dict[
            str,
            dict[str, list[float]],
        ] = {}

        for i in range(n_cuts):
            if pairs is not None:
                part_pairs = pairs.get(
                    part,
                    {},
                )
                level_pairs = part_pairs.get(i)
            else:
                level_pairs = get_pairs_for_level_cuts_(
                    cuts_by_score=cuts_by_score,
                    index=i,
                    codebook=codebook,
                    test_averaged_phones=(test_averaged_phones),
                    levels=levels,
                    global_mean_normalization=(global_mean_normalization),
                )

            if not level_pairs:
                continue

            matrices = create_comparable_matrices(level_pairs)

            for level, level_matrices in matrices.items():
                ref_matrix = level_matrices["ref"]
                test_matrix = level_matrices["test"]

                try:
                    local_cosine_seed = None
                    if cosine_noise_seed is not None:
                        local_cosine_seed = stable_seed(
                            cosine_noise_seed,
                            part,
                            level,
                            i,
                        )
                    distances = compute_all_representation_distances(
                        C_ref=ref_matrix,
                        C_test=test_matrix,
                        C_ref_all=codebook["U"] * codebook["sigma"],
                        maha_mode=maha_mode,
                        maha_alpha=maha_alpha,
                        maha_var_floor=maha_var_floor,
                        maha_gamma=maha_gamma,
                        normalize_by_dim=normalize_by_dim,
                        cosine_noise_mode=cosine_noise_mode,
                        cosine_noise_level=cosine_noise_level,
                        cosine_noise_seed=local_cosine_seed,
                    )
                except (
                    ValueError,
                    np.linalg.LinAlgError,
                ):
                    continue

                if level not in part_distances:
                    part_distances[level] = {metric: [] for metric in distances}

                for metric, value in distances.items():
                    value_array = np.asarray(
                        value,
                        dtype=float,
                    )

                    if not np.all(np.isfinite(value_array)):
                        continue

                    aggregated_value = float(np.mean(value_array))

                    part_distances[level][metric].append(aggregated_value)

        if part_distances:
            overall_distances[part] = part_distances

    return overall_distances







def _score_to_level_name(
    score: float,
    levels: dict[str, float],
) -> str:
    """
    Map a numeric score back to a level name like 'a2', 'b1', ...
    """
    for level_name, level_score in levels.items():
        if np.isclose(float(score), float(level_score)):
            return level_name
    return str(score)


def compute_representation_distances_for_cut(
    codebook: dict[str, Any],
    test_averaged_phones: tuple[np.ndarray, list[Any]],
    global_mean_normalization: bool = True,
    is_averaged: bool = False,
    maha_mode: str = "diag",
    maha_alpha: float = 0.1,
    maha_var_floor: float = 1e-3,
    maha_gamma: float = 0.5,
    normalize_by_dim: bool = True,
    return_common_phones: bool = False,
    cosine_noise_mode: str = "none",
    cosine_noise_level: float = 1.0,
    cosine_noise_seed: int | None = None,
) -> dict[str, Any]:
    """
    Compute representation distances for one test sample.
    """
    X_test, phones_test = test_averaged_phones

    phone_dict_test = {
        phone: i
        for i, phone in enumerate(phones_test)
    }

    pairs = get_pairs_for_test_phones(
        X_test=X_test,
        phone_dict_test=phone_dict_test,
        codebook=codebook,
        global_mean_normalization=global_mean_normalization,
    )


    if not pairs:
        return {}

    matrices = create_comparable_matrices({"cut": pairs})

    if "cut" not in matrices:
        return {}

    ref_matrix = matrices["cut"]["ref"]
    test_matrix = matrices["cut"]["test"]

    if ref_matrix.shape != test_matrix.shape:
        return {}

    if ref_matrix.shape[0] == 0:
        return {}

    try:
        distances = compute_all_representation_distances(
            C_ref=ref_matrix,
            C_test=test_matrix,
            C_ref_all=codebook["U"] * codebook["sigma"],
            maha_mode=maha_mode,
            maha_alpha=maha_alpha,
            maha_var_floor=maha_var_floor,
            maha_gamma=maha_gamma,
            normalize_by_dim=normalize_by_dim,
            cosine_noise_mode=cosine_noise_mode,
            cosine_noise_level=cosine_noise_level,
            cosine_noise_seed=cosine_noise_seed,
        )
    except (ValueError, np.linalg.LinAlgError):
        return {}

    result: dict[str, Any] = {}

    if is_averaged:
        for metric, value in distances.items():
            arr = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(arr)):
                continue
            result[metric] = float(np.mean(arr))
    else:
        for metric, value in distances.items():
            arr = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(arr)):
                continue
            result[metric] = float(arr) if arr.ndim == 0 else arr.copy()

    if not result:
        return {}

    if return_common_phones:
        result["common_phones"] = sorted(pairs)

    return result




def compute_representation_distances_for_recording_id(
    codebook: dict[str, Any],
    recording_id: str,
    test_averaged_phones: dict[str, tuple[np.ndarray, list[Any]]],
    global_mean_normalization: bool = True,
    is_averaged: bool = False,
    maha_mode: str = "diag",
    maha_alpha: float = 0.1,
    maha_var_floor: float = 1e-3,
    maha_gamma: float = 0.5,
    normalize_by_dim: bool = True,
    cosine_noise_mode: str = "none",
    cosine_noise_level: float = 1.0,
    cosine_noise_seed: int = None,
) -> dict[str, Any]:
    """
    Compute distances for one specific recording_id.
    """
    sample = test_averaged_phones.get(recording_id)
    if sample is None:
        return {}

    local_cosine_seed = None
    if cosine_noise_seed is not None:
        local_cosine_seed = stable_seed(
            cosine_noise_seed,
            recording_id,
        )

    return compute_representation_distances_for_cut(
        codebook=codebook,
        test_averaged_phones=sample,
        global_mean_normalization=global_mean_normalization,
        is_averaged=is_averaged,
        maha_mode=maha_mode,
        maha_alpha=maha_alpha,
        maha_var_floor=maha_var_floor,
        maha_gamma=maha_gamma,
        normalize_by_dim=normalize_by_dim,
        return_common_phones=True,
        cosine_noise_mode=cosine_noise_mode,
        cosine_noise_level=cosine_noise_level,
        cosine_noise_seed=local_cosine_seed,
    )



def compute_overall_representation_distances_by_part_per_sample(
    cuts_by_part: dict[str, dict[float, list[tuple[int, Any]]]],
    codebook: dict[str, Any],
    test_averaged_phones: dict[str, tuple[np.ndarray, list[Any]]],
    n_cuts: int = 30,
    levels: dict[str, float] | None = None,
    global_mean_normalization: bool = True,
    maha_mode: str = "diag",
    maha_alpha: float = 0.1,
    maha_var_floor: float = 1e-3,
    maha_gamma: float = 0.5,
    normalize_by_dim: bool = True,
    is_averaged: bool = False,
    cosine_noise_mode: str = "none",
    cosine_noise_level: float = 1.0,
    cosine_noise_seed: int | None = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """
    Compute representation distances independently for each part and each sample.

    Returns
    -------
    dict
        Structure:
        part -> level -> metric -> recording_id -> value

        Additionally:
        part -> level -> "common_phones" -> recording_id -> list[phones]
    """
    if levels is None:
        levels = {
            "a2": 2.0,
            "a2+": 2.5,
            "b1": 3.0,
            "b1+": 3.5,
            "b2": 4.0,
            "b2+": 4.5,
            "c1": 5.0,
        }

    overall_distances: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    for part, cuts_by_score in cuts_by_part.items():
        part_distances: dict[str, dict[str, dict[str, Any]]] = {}

        for score, score_cuts in cuts_by_score.items():
            level = _score_to_level_name(score, levels)

            for _, cut in score_cuts[:n_cuts]:
                recording_id = cut.recording_id

                distances = compute_representation_distances_for_recording_id(
                    codebook=codebook,
                    recording_id=recording_id,
                    test_averaged_phones=test_averaged_phones,
                    global_mean_normalization=global_mean_normalization,
                    is_averaged=is_averaged,
                    maha_mode=maha_mode,
                    maha_alpha=maha_alpha,
                    maha_var_floor=maha_var_floor,
                    maha_gamma=maha_gamma,
                    normalize_by_dim=normalize_by_dim,
                    cosine_noise_mode=cosine_noise_mode,
                    cosine_noise_level=cosine_noise_level,
                    cosine_noise_seed=cosine_noise_seed,
                )

                if not distances:
                    continue

                common_phones = distances.pop("common_phones", None)

                if level not in part_distances:
                    part_distances[level] = {"common_phones": {}}

                if common_phones is not None:
                    part_distances[level]["common_phones"][recording_id] = common_phones

                for metric, value in distances.items():
                    if metric not in part_distances[level]:
                        part_distances[level][metric] = {}

                    part_distances[level][metric][recording_id] = value

        if part_distances:
            overall_distances[part] = part_distances

    return overall_distances




##########################Final Distances Dictionary###############################
def normalize_rank_key(rank: Any) -> Any:
    if isinstance(rank, np.integer):
        return int(rank)

    if isinstance(rank, int):
        return rank

    if isinstance(rank, str):
        try:
            return int(rank)
        except ValueError:
            return rank

    return rank


def find_equivalent_rank_key(
    mapping: dict[Any, Any],
    rank: Any,
) -> Any | None:
    if not isinstance(mapping, dict):
        return None

    candidates: list[Any] = [rank]

    if isinstance(rank, np.integer):
        rank_int = int(rank)
        candidates.extend([rank_int, str(rank_int)])

    elif isinstance(rank, int):
        candidates.extend([np.int64(rank), str(rank)])

    elif isinstance(rank, str):
        try:
            rank_int = int(rank)
            candidates.extend([rank_int, np.int64(rank_int)])
        except ValueError:
            pass

    for candidate in candidates:
        if candidate in mapping:
            return candidate

    return None


def rank_result_is_valid(value: Any) -> bool:
    """
    Existing rank should only be skipped if it contains a non-empty result.
    Empty dict means recompute.
    """
    return isinstance(value, dict) and len(value) > 0


def load_pickle_or_empty(
    path: str | Path,
) -> dict[Any, Any]:
    """
    Load an existing pickle. If it does not exist or contains None,
    return an empty dictionary.
    """
    path = Path(path)

    if not path.exists():
        return {}

    obj = load_pickle(path)

    if obj is None:
        return {}

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected pickle at {path} to contain a dict, "
            f"but got {type(obj)}."
        )

    return obj

def print_available_ranks(
    distances: dict[Any, Any],
) -> None:
    """
    Debug helper: print all currently stored ranks in the nested distance dict.
    """
    for st_key, st_data in distances.items():
        if not isinstance(st_data, dict):
            continue

        for pc_key, pc_data in st_data.items():
            if not isinstance(pc_data, dict):
                continue

            for align_model, align_data in pc_data.items():
                if not isinstance(align_data, dict):
                    continue

                print(
                    f"{st_key} | {pc_key} | {align_model} | ranks:",
                    sorted(align_data.keys(), key=str),
                )




def compute_all_distances_by_part_sandi(
    codebook_task,
    encoder_name,
    layer,
    align_models,
    ranks,
    dataset,
    cuts,
    levels,
    n_samples,
    indices,
    strategies,
    phone_classes,
    skip_combinations,
    codebook_root,
    phone_avg_root,
    params=None,
    calculate_per_sample=False,
    distances=None,
    projection_mode: str = "svd",
    random_seed: int | None = None,
    whiten_power: float = 1.0,
    native_matrix_key: str = "X_ref_all",
    native_phones_key: str = "phones",
    native_matrix_is_centered: bool = False,
    use_global_mean_normalization: bool = True,
):
    current_global_mean_normalization = use_global_mean_normalization

    if params is None:
        params = {
            "maha_mode": "diag",
            "maha_alpha": 0.1,
            "maha_var_floor": 1e-3,
            "maha_gamma": 0.5,
            "normalize_by_dim": True,
            "cosine_noise_mode": "none",
            "cosine_noise_level": 1.0,
            "cosine_noise_seed": None,
        }

    if distances is None:
        distances = {}

    if projection_mode not in {"svd", "random", "whitened", "identity"}:
        raise ValueError(
            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
        )

    ranks = list(ranks)

    print(f"{encoder_name} | layer={layer} | {dataset} started")
    print("Requested ranks:", ranks)
    print("Projection mode:", projection_mode)

    cuts_by_part = collect_sandi_cuts_by_part(
        cuts=cuts,
        upper_bound=n_samples,
        indices=indices,
    )

    for st in strategies:
        for pc in phone_classes:

            if (st, pc) in skip_combinations:
                continue

            st_key = st.value if hasattr(st, "value") else str(st)
            pc_key = pc.value if hasattr(pc, "value") else str(pc)

            distances.setdefault(st_key, {}).setdefault(pc_key, {})

            codebook_path = (
                Path(codebook_root)
                / encoder_name
                / codebook_task
                / f"layer{layer}"
                / st_key
                / f"{pc_key}.pickle"
            )

            codebooks = load_pickle(codebook_path)

            if codebooks is None:
                print(
                    f"WARNING: missing or empty codebook file: {codebook_path}"
                )
                continue

            for align_model in align_models:

                phone_avg_path = (
                    Path(phone_avg_root)
                    / encoder_name
                    / f"layer{layer}"
                    / align_model
                    / dataset
                    / st_key
                    / f"{pc_key}.pickle"
                )

                phone_avgs = load_pickle(phone_avg_path)

                if phone_avgs is None:
                    print(
                        f"WARNING: missing or empty phone averages: {phone_avg_path}"
                    )
                    continue

                distances[st_key][pc_key].setdefault(
                    align_model,
                    {},
                )

                for rank in ranks:
                    rank_store_key = normalize_rank_key(rank)

                    existing_rank_key = find_equivalent_rank_key(
                        mapping=distances[st_key][pc_key][align_model],
                        rank=rank,
                    )

                    # Case 1: rank already exists and has valid data -> skip
                    if existing_rank_key is not None:
                        existing_value = distances[st_key][pc_key][align_model][existing_rank_key]

                        if rank_result_is_valid(existing_value):
                            print(
                                f"Skipping existing valid rank={rank}: "
                                f"{st_key} | {pc_key} | {align_model}"
                            )
                            continue

                        # Case 2: rank exists but is empty/broken -> recompute
                        print(
                            f"Recomputing empty/incomplete rank={rank}: "
                            f"{st_key} | {pc_key} | {align_model}"
                        )
                        del distances[st_key][pc_key][align_model][existing_rank_key]

                    # Case 3: rank is missing from distances -> calculate it
                    codebook_rank_key = find_equivalent_rank_key(
                        mapping=codebooks,
                        rank=rank,
                    )

                    if codebook_rank_key is None:
                        print(
                            f"Cannot calculate rank={rank}, because it is missing "
                            f"from the codebook: {st_key} | {pc_key}"
                        )
                        print(
                            "Available codebook ranks:",
                            sorted(codebooks.keys(), key=str),
                        )
                        continue

                    print(
                        f"Calculating missing rank={rank}: "
                        f"{encoder_name} | layer={layer} | "
                        f"{dataset} | {st_key} | {pc_key} | {align_model} "
                        f"| projection={projection_mode}"
                    )

                    base_codebook = codebooks[codebook_rank_key]

                    if projection_mode == "svd":
                        codebook_for_eval = base_codebook

                    elif projection_mode == "random":
                        codebook_for_eval = make_random_projection_codebook(
                            base_codebook=base_codebook,
                            rank=len(np.asarray(base_codebook["sigma"]).ravel()),
                            random_state=random_seed,
                            native_matrix_key=native_matrix_key,
                            native_phones_key=native_phones_key,
                            native_matrix_is_centered=native_matrix_is_centered,
                        )

                    elif projection_mode == "whitened":
                        codebook_for_eval = make_power_whitened_codebook(
                            base_codebook=base_codebook,
                            whiten_power=whiten_power,
                            rank=len(np.asarray(base_codebook["sigma"]).ravel()),
                        )

                    elif projection_mode == "identity":
                        codebook_for_eval = make_identity_codebook(
                            base_codebook=base_codebook,
                            native_matrix_key=native_matrix_key,
                            native_phones_key=native_phones_key,
                        )
                        current_global_mean_normalization = False

                    else:
                        raise ValueError(
                            "projection_mode must be one of: 'svd', "
                            "'random', 'whitened', 'identity'."
                        )


                    calculator = (
                        compute_overall_representation_distances_by_part_per_sample
                        if calculate_per_sample
                        else compute_overall_representation_distances_by_part
                    )

                    rank_result = calculator(
                        cuts_by_part=cuts_by_part,
                        levels=levels,
                        codebook=codebook_for_eval,
                        test_averaged_phones=phone_avgs,
                        n_cuts=n_samples,
                        global_mean_normalization=current_global_mean_normalization,
                        **params,
                    )

                    if not rank_result_is_valid(rank_result):
                        print(
                            f"WARNING: rank={rank} produced an empty result. "
                            f"Not storing it."
                        )
                        continue

                    distances[st_key][pc_key][align_model][rank_store_key] = rank_result

    return distances




def compute_and_save_distances_sandi(
    codebook_task,
    models,
    align_models,
    ranks,
    datasets,
    indices,
    cuts,
    levels,
    n_samples,
    strategies,
    phone_classes,
    skip_combinations,
    codebook_root,
    phone_avg_root,
    output_dir,
    params=None,
    calculate_per_sample=False,
    projection_mode: str = "svd",
    random_seeds: list[int] | None = None,
    whiten_power: float = 1.0,
    overwrite: bool = False,
    native_matrix_key: str = "X_ref_all",
    native_phones_key: str = "phones",
    native_matrix_is_centered: bool = False,
):
    output_dir = Path(output_dir)

    if projection_mode not in {"svd", "random", "whitened", "identity"}:
        raise ValueError(
            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
        )

    if projection_mode == "random":
        if random_seeds is None:
            random_seeds = [0]
        seeds_to_run = list(random_seeds)
    else:
        seeds_to_run = [None]

    for encoder_name, info in models.items():
        for layer in info["layer"]:
            for dataset in datasets:
                dataset_indices = None if indices is None else indices[dataset]

                for seed in seeds_to_run:
                    if projection_mode == "svd":
                        projection_folder = f"{codebook_task}_codebook"
                        seed_folder = ""
                    elif projection_mode == "random":
                        projection_folder = f"{codebook_task}_random_projection_codebook"
                        seed_folder = f"/seed{seed}"
                    elif projection_mode == "whitened":
                        projection_folder = f"{codebook_task}_whitened_p{whiten_power}_codebook"
                        seed_folder = ""
                    else:
                        projection_folder = f"{codebook_task}_identity_codebook"

                    save_path = (
                        output_dir
                        / projection_folder
                        / "sandi"
                        / encoder_name
                        / f"layer{layer}"
                    )

                    if projection_mode == "random":
                        save_path = save_path / f"seed{seed}"

                    noise_suffix = make_cosine_noise_suffix(params)

                    save_path = save_path / (
                        f"{dataset}{'_sample' if calculate_per_sample else ''}{noise_suffix}.pickle"
                    )

                    print(save_path)

                    if save_path.exists() and not overwrite:
                        print(f"Already calculated: {save_path}")
                        continue

                    distances = compute_all_distances_by_part_sandi(
                        codebook_task=codebook_task,
                        encoder_name=encoder_name,
                        layer=layer,
                        align_models=align_models,
                        ranks=ranks,
                        dataset=dataset,
                        cuts=cuts[dataset],
                        levels=levels,
                        n_samples=n_samples,
                        indices=dataset_indices,
                        strategies=strategies,
                        phone_classes=phone_classes,
                        skip_combinations=skip_combinations,
                        codebook_root=codebook_root,
                        phone_avg_root=phone_avg_root,
                        params=params,
                        calculate_per_sample=calculate_per_sample,
                        distances=None,
                        projection_mode=projection_mode,
                        random_seed=seed,
                        whiten_power=whiten_power,
                        native_matrix_key=native_matrix_key,
                        native_phones_key=native_phones_key,
                        native_matrix_is_centered=native_matrix_is_centered,
                    )

                    save_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    save_pickle(distances, save_path)
                    print(f"Saved: {save_path}")


##################### UME-ERJ ####################################################
###############################################################################
# UME-ERJ: distance calculation by individual rater
###############################################################################
##################### UME-ERJ ####################################################
###############################################################################
# UME-ERJ: distance calculation by individual rater
###############################################################################
# ---------------------------------------------------------------------
# Expected external functions/constants from your project:
#
# - load_pickle(path)
# - save_pickle(obj, path)
# - compute_representation_distances_for_recording_id(...)
# - UMEERJ_INVALID_SCORES
#
# compute_representation_distances_for_recording_id should return:
# {
#     "d_euclidean_per_phone": np.ndarray,
#     "D_frobenius": float,
#     "d_cosine_per_phone": np.ndarray,
#     "D_procrustes": float,
#     "d_mahalanobis_per_phone": np.ndarray,
#     "common_phones": list[Any],  # optional
# }
# ---------------------------------------------------------------------


# =====================================================================
# Generic helpers
# =====================================================================

def umeerj_get_key(
        value: Any,
) -> str:
    """
    Return enum .value if available, otherwise string representation.
    """
    return value.value if hasattr(value, "value") else str(value)


def umeerj_filter_kwargs(
        kwargs: dict[str, Any],
        allowed_keys: set[str],
) -> dict[str, Any]:
    """
    Keep only kwargs whose keys are in allowed_keys.
    """
    return {
        key: value
        for key, value in kwargs.items()
        if key in allowed_keys
    }


def umeerj_should_skip_combination(
        strategy: Any,
        phone_class: Any,
        skip_combinations: set[tuple[Any, Any]] | list[tuple[Any, Any]],
) -> bool:
    """
    Check whether a strategy/phone-class combination should be skipped.

    Supports both enum objects and their string keys.
    """
    strategy_key = umeerj_get_key(strategy)
    phone_class_key = umeerj_get_key(phone_class)

    return (
        (strategy, phone_class) in skip_combinations
        or (strategy_key, phone_class_key) in skip_combinations
    )


def umeerj_distance_to_scalar(
        value: Any,
        aggregation: str = "mean",
        trim_ratio: float = 0.1,
) -> float:
    """
    Convert a raw distance value to one scalar.

    Scalar values are returned unchanged.
    Vector values, e.g. per-phone distances, are aggregated.

    Parameters
    ----------
    value
        Scalar or array-like distance value.

    aggregation
        Aggregation strategy for vector values.

        Supported:
            - "mean"
            - "median"
            - "trimmed_mean"
            - "max"
            - "min"

    trim_ratio
        Used only when aggregation == "trimmed_mean".
        Example: 0.1 removes the lowest 10% and highest 10%.

    Returns
    -------
    float
        Aggregated scalar distance, or np.nan if invalid.
    """
    arr = np.asarray(value, dtype=float)

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


def umeerj_clean_distance_dict(
        distances: dict[str, Any],
        distance_storage: str = "raw",
        phone_aggregation: str = "mean",
        keep_common_phones: bool = True,
) -> dict[str, Any]:
    """
    Clean and optionally aggregate the output of
    compute_representation_distances_for_recording_id.

    Parameters
    ----------
    distances
        Raw distance dictionary for one recording.

    distance_storage
        "raw"
            Store scalar metrics as floats and phone-level metrics as arrays.

        "scalar"
            Convert every metric to one scalar using phone_aggregation.

    phone_aggregation
        Aggregation used only if distance_storage == "scalar".

    keep_common_phones
        Whether to retain "common_phones" in the saved output.

    Returns
    -------
    dict[str, Any]
        Cleaned distance dictionary.
    """
    if distance_storage not in {"raw", "scalar"}:
        raise ValueError("distance_storage must be either 'raw' or 'scalar'.")

    cleaned: dict[str, Any] = {}

    for metric, value in distances.items():
        if metric == "common_phones":
            if keep_common_phones:
                cleaned[metric] = value
            continue

        arr = np.asarray(value, dtype=float)

        if arr.size == 0:
            continue

        if not np.all(np.isfinite(arr)):
            continue

        if distance_storage == "scalar":
            scalar = umeerj_distance_to_scalar(
                value=value,
                aggregation=phone_aggregation,
            )

            if np.isfinite(scalar):
                cleaned[metric] = float(scalar)

        else:
            if arr.ndim == 0:
                cleaned[metric] = float(arr)
            else:
                cleaned[metric] = arr.copy()

    return cleaned


# =====================================================================
# Recording selection
# =====================================================================

def umeerj_iter_selected_cuts(
        cuts: Iterable[Any],
        indices: set[int] | None = None,
        max_recordings: int | None = None,
) -> list[tuple[int, Any]]:
    """
    Select cuts by index and return unique recording IDs in dataset order.

    Notes
    -----
    Distance calculation should be label-independent, so this function
    does not look at rater scores or levels.
    """
    selected: list[tuple[int, Any]] = []
    seen_recordings: set[str] = set()

    for idx, cut in enumerate(cuts):
        if indices is not None and idx not in indices:
            continue

        recording_id = str(cut.recording_id)

        if recording_id in seen_recordings:
            continue

        selected.append((idx, cut))
        seen_recordings.add(recording_id)

        if max_recordings is not None and len(selected) >= max_recordings:
            break

    return selected


# =====================================================================
# Core distance calculation by recording
# =====================================================================

def umeerj_compute_distances_by_recording(
        cuts: Iterable[Any],
        codebook: dict[str, Any],
        test_averaged_phones: dict[str, tuple[np.ndarray, list[Any]]],
        indices: set[int] | None = None,
        max_recordings: int | None = None,
        global_mean_normalization: bool = True,
        maha_mode: str = "diag",
        maha_alpha: float = 0.1,
        maha_var_floor: float = 1e-3,
        maha_gamma: float = 0.5,
        normalize_by_dim: bool = True,
        distance_storage: str = "raw",
        phone_aggregation: str = "mean",
        keep_common_phones: bool = True,
        cache_distances: bool = True,
        cosine_noise_mode: str = "none",
        cosine_noise_level: float = 1.0,
        cosine_noise_seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Compute UME-ERJ distances once per recording.

    This function is intentionally independent of rater scores.

    Returns
    -------
    dict
        Structure:
            recording_id -> metric -> value

        If distance_storage == "raw":
            - scalar metrics are floats
            - phone-level metrics are np.ndarray

        If distance_storage == "scalar":
            - all metrics are floats

        Optionally:
            recording_id -> "common_phones" -> list[phones]
    """
    selected_cuts = umeerj_iter_selected_cuts(
        cuts=cuts,
        indices=indices,
        max_recordings=max_recordings,
    )

    result: dict[str, dict[str, Any]] = {}
    recording_cache: dict[str, dict[str, Any]] = {}

    for _, cut in tqdm(
            selected_cuts,
            desc="UME-ERJ recording distances",
    ):
        recording_id = str(cut.recording_id)

        if recording_id in result:
            continue

        if cache_distances and recording_id in recording_cache:
            distances = dict(recording_cache[recording_id])
        else:
            distances = compute_representation_distances_for_recording_id(
                codebook=codebook,
                recording_id=recording_id,
                test_averaged_phones=test_averaged_phones,
                global_mean_normalization=global_mean_normalization,

                # Important:
                # Always request raw values here.
                # Aggregation, if wanted, is handled by umeerj_clean_distance_dict.
                is_averaged=False,

                maha_mode=maha_mode,
                maha_alpha=maha_alpha,
                maha_var_floor=maha_var_floor,
                maha_gamma=maha_gamma,
                normalize_by_dim=normalize_by_dim,
                cosine_noise_mode=cosine_noise_mode,
                cosine_noise_level=cosine_noise_level,
                cosine_noise_seed=cosine_noise_seed,
            )

            if cache_distances:
                recording_cache[recording_id] = dict(distances)

        if not distances:
            continue

        cleaned = umeerj_clean_distance_dict(
            distances=distances,
            distance_storage=distance_storage,
            phone_aggregation=phone_aggregation,
            keep_common_phones=keep_common_phones,
        )

        metric_keys = [
            key
            for key in cleaned
            if key != "common_phones"
        ]

        if not metric_keys:
            continue

        result[recording_id] = cleaned

    return result


# =====================================================================
# Full grid calculation
# =====================================================================


def umeerj_compute_all_distance_grid_by_recording(
    codebook_task: str,
    encoder_name: str,
    layer: int,
    align_models: Iterable[str],
    ranks: Iterable[Any],
    dataset: str,
    cuts: Iterable[Any],
    indices: set[int] | None,
    strategies: Iterable[Any],
    phone_classes: Iterable[Any],
    skip_combinations: set[tuple[Any, Any]] | list[tuple[Any, Any]],
    codebook_root: str | Path,
    phone_avg_root: str | Path,
    params: dict[str, Any] | None = None,
    max_recordings: int | None = None,
    distances: dict | None = None,
    projection_mode: str = "svd",
    random_seed: int | None = None,
    whiten_power: float = 1.0,
    native_matrix_key: str = "X_ref_all",
    native_phones_key: str = "phones",
    native_matrix_is_centered: bool = False,
    use_global_mean_normalization: bool = True,
) -> dict:
    if params is None:
        params = {
            "maha_mode": "diag",
            "maha_alpha": 0.1,
            "maha_var_floor": 1e-3,
            "maha_gamma": 0.5,
            "normalize_by_dim": True,
            "cosine_noise_mode": "none",
            "cosine_noise_level": 1.0,
            "cosine_noise_seed": None,
            "distance_storage": "raw",
            "phone_aggregation": "mean",
            "keep_common_phones": True,
            "cache_distances": True,
        }

    params = dict(params)
    current_global_mean_normalization = use_global_mean_normalization

    distance_kwargs = umeerj_filter_kwargs(
        kwargs=params,
        allowed_keys={
            "maha_mode",
            "maha_alpha",
            "maha_var_floor",
            "maha_gamma",
            "normalize_by_dim",
            "distance_storage",
            "phone_aggregation",
            "keep_common_phones",
            "cache_distances",
            "cosine_noise_mode",
            "cosine_noise_level",
            "cosine_noise_seed",
        },
    )

    if distances is None:
        distances = {}

    if projection_mode not in {"svd", "random", "whitened", "identity"}:
        raise ValueError(
            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
        )

    ranks = list(ranks)

    print(
        f"{encoder_name} | layer={layer} | {dataset} | "
        f"UME-ERJ recording-level distance calculation started"
    )
    print("Requested ranks:", ranks)
    print("Projection mode:", projection_mode)

    for st in strategies:
        for pc in phone_classes:

            if umeerj_should_skip_combination(
                strategy=st,
                phone_class=pc,
                skip_combinations=skip_combinations,
            ):
                continue

            st_key = umeerj_get_key(st)
            pc_key = umeerj_get_key(pc)

            distances.setdefault(st_key, {}).setdefault(pc_key, {})

            codebook_path = (
                Path(codebook_root)
                / encoder_name
                / codebook_task
                / f"layer{layer}"
                / st_key
                / f"{pc_key}.pickle"
            )

            codebooks = load_pickle(codebook_path)

            if codebooks is None:
                print(f"WARNING: missing or empty codebook file: {codebook_path}")
                continue

            for align_model in align_models:
                phone_avg_path = (
                    Path(phone_avg_root)
                    / encoder_name
                    / f"layer{layer}"
                    / align_model
                    / dataset
                    / st_key
                    / f"{pc_key}.pickle"
                )

                phone_avgs = load_pickle(phone_avg_path)

                if phone_avgs is None:
                    print(f"WARNING: missing or empty phone averages: {phone_avg_path}")
                    continue

                distances[st_key][pc_key].setdefault(align_model, {})

                for rank in ranks:
                    rank_store_key = normalize_rank_key(rank)

                    existing_rank_key = find_equivalent_rank_key(
                        mapping=distances[st_key][pc_key][align_model],
                        rank=rank,
                    )

                    if existing_rank_key is not None:
                        existing_value = distances[st_key][pc_key][align_model][existing_rank_key]

                        if rank_result_is_valid(existing_value):
                            print(
                                f"Skipping existing valid rank={rank}: "
                                f"{st_key} | {pc_key} | {align_model}"
                            )
                            continue

                        print(
                            f"Recomputing empty/incomplete rank={rank}: "
                            f"{st_key} | {pc_key} | {align_model}"
                        )
                        del distances[st_key][pc_key][align_model][existing_rank_key]

                    codebook_rank_key = find_equivalent_rank_key(
                        mapping=codebooks,
                        rank=rank,
                    )

                    if codebook_rank_key is None:
                        print(
                            f"Cannot calculate rank={rank}, because it is missing "
                            f"from the codebook: {st_key} | {pc_key}"
                        )
                        print("Available codebook ranks:", sorted(codebooks.keys(), key=str))
                        continue

                    print(
                        f"Calculating missing rank={rank}: "
                        f"{st_key} | {pc_key} | {align_model} | "
                        f"projection={projection_mode}"
                    )

                    base_codebook = codebooks[codebook_rank_key]

                    if projection_mode == "svd":
                        codebook_for_eval = base_codebook

                    elif projection_mode == "random":
                        codebook_for_eval = make_random_projection_codebook(
                            base_codebook=base_codebook,
                            rank=len(np.asarray(base_codebook["sigma"]).ravel()),
                            random_state=random_seed,
                            native_matrix_key=native_matrix_key,
                            native_phones_key=native_phones_key,
                            native_matrix_is_centered=native_matrix_is_centered,
                        )

                    elif projection_mode == "whitened":
                        codebook_for_eval = make_power_whitened_codebook(
                            base_codebook=base_codebook,
                            whiten_power=whiten_power,
                            rank=len(np.asarray(base_codebook["sigma"]).ravel()),
                        )

                    elif projection_mode == "identity":
                        codebook_for_eval = make_identity_codebook(
                            base_codebook=base_codebook,
                            native_matrix_key=native_matrix_key,
                            native_phones_key=native_phones_key,
                        )
                        current_global_mean_normalization = False

                    else:
                        raise ValueError(
                            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
                        )

                    calculator = (
                        umeerj_compute_distances_by_recording
                    )

                    rank_result = calculator(
                        cuts=cuts,
                        indices=indices,
                        max_recordings=max_recordings,
                        codebook=codebook_for_eval,
                        test_averaged_phones=phone_avgs,
                        global_mean_normalization=current_global_mean_normalization,
                        **distance_kwargs,
                    )

                    if not rank_result_is_valid(rank_result):
                        print(
                            f"WARNING: rank={rank} produced an empty result. "
                            f"Not storing it."
                        )
                        continue

                    distances[st_key][pc_key][align_model][rank_store_key] = rank_result

    return distances
# =====================================================================
# Save wrapper
# =====================================================================

def umeerj_make_distance_save_path(
    output_dir: str | Path,
    codebook_task: str,
    encoder_name: str,
    layer: int,
    dataset: str,
    params: dict[str, Any] | None = None,
    projection_mode: str = "svd",
    random_seed: int | None = None,
    whiten_power: float = 1.0,
) -> Path:
    """
    Build a save path for UME-ERJ distances that also encodes the
    projection mode.

    projection_mode:
        "svd"
        "random"
        "whitened"
    """
    output_dir = Path(output_dir)

    params = {} if params is None else dict(params)

    distance_storage = params.get("distance_storage", "raw")
    phone_aggregation = params.get("phone_aggregation", "mean")

    if distance_storage == "raw":
        suffix = "raw"
    elif distance_storage == "scalar":
        suffix = f"scalar_{phone_aggregation}"
    else:
        raise ValueError("distance_storage must be either 'raw' or 'scalar'.")
    suffix = suffix + make_cosine_noise_suffix(params)

    if projection_mode == "svd":
        proj_folder = "svd"
    elif projection_mode == "random":
        proj_folder = f"random_seed{random_seed}"
    elif projection_mode == "whitened":
        proj_folder = f"whitened_p{whiten_power}"
    elif projection_mode == "identity":
        proj_folder = "identity"
    else:
        raise ValueError(
            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
        )

    return (
        output_dir
        / f"{codebook_task}_codebook"
        / "ume_erj"
        / "by_recording"
        / proj_folder
        / encoder_name
        / f"layer{layer}"
        / f"{dataset}_{suffix}.pickle"
    )



def umeerj_compute_and_save_distance_grid_by_recording(
    codebook_task: str,
    models: dict[str, Any],
    align_models: Iterable[str],
    ranks: Iterable[Any],
    datasets: Iterable[str],
    indices: dict[str, set[int]] | None,
    cuts: dict[str, Iterable[Any]],
    strategies: Iterable[Any],
    phone_classes: Iterable[Any],
    skip_combinations: set[tuple[Any, Any]] | list[tuple[Any, Any]],
    codebook_root: str | Path,
    phone_avg_root: str | Path,
    output_dir: str | Path,
    params: dict[str, Any] | None = None,
    max_recordings: int | None = None,
    overwrite: bool = False,
    projection_mode: str = "svd",
    random_seeds: list[int] | None = None,
    whiten_power: float = 1.0,
    native_matrix_key: str = "X_ref_all",
    native_phones_key: str = "phones",
    native_matrix_is_centered: bool = False,
) -> None:
    output_dir = Path(output_dir)

    if projection_mode not in {"svd", "random", "whitened", "identity"}:
        raise ValueError(
            "projection_mode must be one of: 'svd', 'random', 'whitened', 'identity'."
        )

    if projection_mode == "random":
        if random_seeds is None:
            random_seeds = [0]
        seeds_to_run = list(random_seeds)
    else:
        seeds_to_run = [None]

    for encoder_name, info in models.items():
        for layer in info["layer"]:
            for dataset in datasets:
                dataset_indices = None if indices is None else indices[dataset]

                for seed in seeds_to_run:
                    save_path = umeerj_make_distance_save_path(
                        output_dir=output_dir,
                        codebook_task=codebook_task,
                        encoder_name=encoder_name,
                        layer=layer,
                        dataset=dataset,
                        params=params,
                        projection_mode=projection_mode,
                        random_seed=seed,
                        whiten_power=whiten_power,
                    )

                    print(save_path)

                    if save_path.exists() and not overwrite:
                        print(f"Already calculated: {save_path}")
                        continue

                    distances = umeerj_compute_all_distance_grid_by_recording(
                        codebook_task=codebook_task,
                        encoder_name=encoder_name,
                        layer=layer,
                        align_models=align_models,
                        ranks=ranks,
                        dataset=dataset,
                        cuts=cuts[dataset],
                        indices=dataset_indices,
                        strategies=strategies,
                        phone_classes=phone_classes,
                        skip_combinations=skip_combinations,
                        codebook_root=codebook_root,
                        phone_avg_root=phone_avg_root,
                        params=params,
                        max_recordings=max_recordings,
                        projection_mode=projection_mode,
                        random_seed=seed,
                        whiten_power=whiten_power,
                        native_matrix_key=native_matrix_key,
                        native_phones_key=native_phones_key,
                        native_matrix_is_centered=native_matrix_is_centered,
                    )

                    save_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    save_pickle(distances, save_path)

                    print(f"Saved: {save_path}")