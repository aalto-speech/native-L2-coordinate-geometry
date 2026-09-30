from typing import Any
import numpy as np
from pathlib import Path


def normalize_phone_label(
    phone: str | tuple[str, ...],
) -> str | tuple[str, ...]:
    """
    Normalize phone/n-phone labels for dictionary lookup.
    """
    if isinstance(phone, tuple):
        return tuple(str(element).upper() for element in phone)
    return str(phone).upper()


def random_orthonormal_projection(
    n_features: int,
    rank: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Create Q in R^{F x K} with orthonormal columns.
    """
    if rank > n_features:
        raise ValueError(
            f"rank={rank} cannot exceed n_features={n_features}."
        )

    rng = np.random.default_rng(random_state)
    A = rng.standard_normal(size=(n_features, rank))
    Q, R = np.linalg.qr(A, mode="reduced")

    # Fix QR sign ambiguity for reproducibility.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[None, :]

    return Q


def align_matrix_to_phone_order(
    X: np.ndarray,
    source_phones: list[Any],
    target_phones: list[Any],
) -> np.ndarray:
    """
    Reorder rows of X so they follow target_phones.
    Assumes source_phones[i] corresponds to X[i].
    """
    X = np.asarray(X, dtype=np.float64)

    source_dict = {
        normalize_phone_label(phone): i
        for i, phone in enumerate(source_phones)
    }

    rows = []
    missing = []

    for phone in target_phones:
        norm_phone = normalize_phone_label(phone)

        if norm_phone not in source_dict:
            missing.append(phone)
            continue

        rows.append(X[source_dict[norm_phone]])

    if missing:
        raise ValueError(
            f"{len(missing)} phones from target_phones are missing in "
            f"source_phones. Examples: {missing[:10]}"
        )

    return np.stack(rows, axis=0)


def make_random_projection_codebook(
    base_codebook: dict[str, Any],
    rank: int | None = None,
    random_state: int | None = None,
    native_matrix_key: str = "X_ref_all",
    native_phones_key: str = "phones",
    native_matrix_is_centered: bool = False,
) -> dict[str, Any]:
    """
    Create a random-projection codebook compatible with your existing code.

    Required:
        base_codebook[native_matrix_key]
            Native averaged feature matrix before SVD.
        base_codebook["phones"]
            Phone/n-phone labels in codebook order.
        base_codebook["global_mean"]
            Native global mean in original feature space.

    The returned codebook works with:
        c_ref = codebook["U"] * codebook["sigma"]
        transformed_test = get_transformed(X_test, codebook["VT"])
    """
    if native_matrix_key not in base_codebook:
        raise KeyError(
            f"Missing key '{native_matrix_key}' in base_codebook. "
            f"Random projection needs the original native averaged matrix."
        )

    X_native = np.asarray(base_codebook[native_matrix_key], dtype=np.float64)

    codebook_phones = list(base_codebook["phones"])
    native_phones = list(base_codebook.get(native_phones_key, codebook_phones))

    if len(native_phones) != X_native.shape[0]:
        raise ValueError(
            f"len(native_phones)={len(native_phones)} but "
            f"X_native has {X_native.shape[0]} rows."
        )

    X_native = align_matrix_to_phone_order(
        X=X_native,
        source_phones=native_phones,
        target_phones=codebook_phones,
    )

    global_mean = np.asarray(base_codebook["global_mean"], dtype=np.float64)

    if rank is None:
        rank = len(np.asarray(base_codebook["sigma"]).ravel())

    if rank > X_native.shape[1]:
        raise ValueError(
            f"rank={rank} cannot exceed original feature dimension "
            f"{X_native.shape[1]}."
        )

    Q = random_orthonormal_projection(
        n_features=X_native.shape[1],
        rank=rank,
        random_state=random_state,
    )

    if native_matrix_is_centered:
        X_centered = X_native
    else:
        X_centered = X_native - global_mean

    C_native_random = X_centered @ Q

    random_codebook = dict(base_codebook)
    random_codebook["U"] = C_native_random
    random_codebook["sigma"] = np.ones(rank, dtype=np.float64)
    random_codebook["VT"] = Q.T
    random_codebook["phones"] = codebook_phones
    random_codebook["global_mean"] = global_mean
    random_codebook["projection_type"] = "random_orthonormal"
    random_codebook["random_state"] = random_state
    random_codebook["actual_rank"] = rank

    return random_codebook



def make_power_whitened_codebook(
    base_codebook: dict[str, Any],
    whiten_power: float = 1.0,
    rank: int | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Create a whitened / power-whitened SVD codebook compatible with the
    existing distance code.

    The transformed coordinates are:

        C^(p) = U * sigma^(1-p) = (X - mu) V sigma^(-p)

    where:
        p = 0   -> original SVD coordinates
        p = 1   -> full whitening
        p = 0.5 -> partial whitening
    """
    U = np.asarray(base_codebook["U"], dtype=np.float64)
    sigma = np.asarray(base_codebook["sigma"], dtype=np.float64).ravel()
    VT = np.asarray(base_codebook["VT"], dtype=np.float64)

    if rank is None:
        rank = len(sigma)

    if rank > len(sigma):
        raise ValueError(
            f"Requested rank={rank}, but sigma only has length {len(sigma)}."
        )

    U = U[:, :rank]
    sigma = sigma[:rank]
    VT = VT[:rank, :]

    sigma_safe = np.maximum(sigma, eps)

    ref_scale = sigma_safe ** (1.0 - whiten_power)
    proj_scale = sigma_safe ** (-whiten_power)

    VT_whitened = proj_scale[:, None] * VT

    whitened_codebook = dict(base_codebook)
    whitened_codebook["U"] = U
    whitened_codebook["sigma"] = ref_scale
    whitened_codebook["VT"] = VT_whitened
    whitened_codebook["projection_type"] = "power_whitened_svd"
    whitened_codebook["whiten_power"] = whiten_power
    whitened_codebook["actual_rank"] = rank

    return whitened_codebook



def apply_white_noise_for_cosine(
    C_ref: np.ndarray,
    C_test: np.ndarray,
    mode: str = "none",
    noise_level: float = 1.0,
    random_state: int | np.random.Generator | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply white-noise control before cosine distance.

    mode:
        "none"
            No noise.

        "replace_independent"
            Replace both C_ref and C_test by independent white noise.
            This is the strongest negative control.

        "add_independent"
            Add independent white noise to both C_ref and C_test.

        "add_test"
            Add white noise only to C_test.

    noise_level:
        For additive modes, this is relative to row RMS.
        noise_level=1.0 means noise RMS approximately equals vector RMS.
    """
    C_ref = np.asarray(C_ref, dtype=np.float64)
    C_test = np.asarray(C_test, dtype=np.float64)

    if C_ref.shape != C_test.shape:
        raise ValueError("C_ref and C_test must have the same shape.")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if mode == "none":
        return C_ref, C_test

    if mode == "replace_independent":
        # Scale does not matter much for cosine, but keeping similar scale
        # avoids numerical issues.
        ref_scale = np.sqrt(np.mean(C_ref**2))
        test_scale = np.sqrt(np.mean(C_test**2))

        ref_scale = max(float(ref_scale), eps)
        test_scale = max(float(test_scale), eps)

        C_ref_noise = rng.normal(
            loc=0.0,
            scale=ref_scale,
            size=C_ref.shape,
        )

        C_test_noise = rng.normal(
            loc=0.0,
            scale=test_scale,
            size=C_test.shape,
        )

        return C_ref_noise, C_test_noise

    # Row-wise RMS scaling for additive noise.
    ref_row_scale = np.sqrt(
        np.mean(C_ref**2, axis=1, keepdims=True)
    )
    test_row_scale = np.sqrt(
        np.mean(C_test**2, axis=1, keepdims=True)
    )

    ref_row_scale = np.maximum(ref_row_scale, eps)
    test_row_scale = np.maximum(test_row_scale, eps)

    if mode == "add_independent":
        C_ref_noisy = C_ref + noise_level * ref_row_scale * rng.normal(
            size=C_ref.shape
        )
        C_test_noisy = C_test + noise_level * test_row_scale * rng.normal(
            size=C_test.shape
        )

        return C_ref_noisy, C_test_noisy

    if mode == "add_test":
        C_test_noisy = C_test + noise_level * test_row_scale * rng.normal(
            size=C_test.shape
        )

        return C_ref, C_test_noisy

    raise ValueError(
        "mode must be one of: "
        "'none', 'replace_independent', 'add_independent', 'add_test'."
    )