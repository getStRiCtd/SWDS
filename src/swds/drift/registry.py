from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from time import perf_counter
from typing import Callable

from swds.drift.c2st import classifier_two_sample_score
from swds.drift.energy import energy_distance_score
from swds.drift.ks import max_ks_score, mean_ks_score
from swds.drift.mmd import mmd_rbf_score
from swds.drift.psi import max_psi_score, mean_psi_score
from swds.drift.sinkhorn import sinkhorn_divergence_score
from swds.drift.sliced_wasserstein import sliced_wasserstein_score


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftScore:
    method: str
    score: float
    runtime_seconds: float


DEFAULT_METHODS = (
    "swds",
    "mean_ks",
    "max_ks",
    "mean_psi",
    "max_psi",
    "mmd_rbf",
    "energy",
    "c2st",
)

OPTIONAL_METHODS = ("sinkhorn",)


def drift_method_registry(seed: int = 42) -> dict[str, Callable]:
    return {
        "swds": lambda a, b: sliced_wasserstein_score(a, b, seed=seed),
        "mean_ks": lambda a, b: mean_ks_score(a, b, seed=seed),
        "max_ks": lambda a, b: max_ks_score(a, b, seed=seed),
        "mean_psi": lambda a, b: mean_psi_score(a, b, seed=seed),
        "max_psi": lambda a, b: max_psi_score(a, b, seed=seed),
        "mmd_rbf": lambda a, b: mmd_rbf_score(a, b, seed=seed),
        "energy": lambda a, b: energy_distance_score(a, b, seed=seed),
        "c2st": lambda a, b: classifier_two_sample_score(a, b, seed=seed),
        "sinkhorn": lambda a, b: sinkhorn_divergence_score(a, b, seed=seed),
    }


def compute_drift_scores(
    X_ref,
    X_cur,
    *,
    methods: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
) -> list[DriftScore]:
    methods = list(methods or DEFAULT_METHODS)
    registry = drift_method_registry(seed=seed)
    LOGGER.debug(
        "computing drift scores methods=%s seed=%d ref_shape=%s cur_shape=%s",
        methods,
        seed,
        tuple(getattr(X_ref, "shape", ())),
        tuple(getattr(X_cur, "shape", ())),
    )

    results: list[DriftScore] = []
    for method in methods:
        scorer = registry.get(method) or _dynamic_method(method, seed=seed)
        if scorer is None:
            LOGGER.error("unknown drift method requested method=%s", method)
            raise ValueError(f"unknown drift method: {method!r}")
        LOGGER.debug("drift method started method=%s", method)
        started = perf_counter()
        score = float(scorer(X_ref, X_cur))
        elapsed = perf_counter() - started
        LOGGER.debug("drift method completed method=%s score=%.8f runtime=%.4fs", method, score, elapsed)
        results.append(DriftScore(method=method, score=score, runtime_seconds=elapsed))
    return results


def _dynamic_method(method: str, *, seed: int) -> Callable | None:
    match = re.fullmatch(r"swds(?:_k(?P<k>\d+))?(?:_q(?P<q>\d+))?", method)
    if not match or method == "swds":
        return None
    n_projections = int(match.group("k") or 128)
    n_quantiles = int(match.group("q") or 512)
    LOGGER.debug("using dynamic SWDS method method=%s n_projections=%d n_quantiles=%d", method, n_projections, n_quantiles)
    return lambda a, b: sliced_wasserstein_score(
        a,
        b,
        n_projections=n_projections,
        n_quantiles=n_quantiles,
        seed=seed,
    )
