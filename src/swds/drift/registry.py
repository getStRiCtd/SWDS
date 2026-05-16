from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from time import perf_counter
from typing import Callable

from joblib import Parallel, delayed

from swds.drift.c2st import classifier_two_sample_score
from swds.drift.energy import energy_distance_score
from swds.drift.ks import max_ks_score, mean_ks_score
from swds.drift.mmd import mmd_rbf_score
from swds.drift.psi import max_psi_score, mean_psi_score
from swds.drift.sinkhorn import sinkhorn_divergence_score
from swds.drift.sliced_wasserstein import (
    prepare_sliced_wasserstein_reference,
    random_directions,
    score_sliced_wasserstein_prepared,
    sliced_wasserstein_score,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftScore:
    method: str
    score: float
    runtime_seconds: float


@dataclass(frozen=True)
class DriftRuntimeConfig:
    seed: int = 42
    n_jobs: int = 1
    swds_backend: str = "auto"
    swds_device: str | None = None
    mmd_max_samples: int = 1000
    energy_max_samples: int = 1000
    c2st_max_samples: int = 4000
    c2st_n_splits: int = 5
    c2st_n_jobs: int | None = None


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


def drift_method_registry(
    *,
    seed: int = 42,
    swds_backend: str = "auto",
    swds_device: str | None = None,
    mmd_max_samples: int = 1000,
    energy_max_samples: int = 1000,
    c2st_max_samples: int = 4000,
    c2st_n_splits: int = 5,
    c2st_n_jobs: int | None = None,
) -> dict[str, Callable]:
    return {
        "swds": lambda a, b: sliced_wasserstein_score(a, b, seed=seed, backend=swds_backend, device=swds_device),
        "mean_ks": lambda a, b: mean_ks_score(a, b, seed=seed),
        "max_ks": lambda a, b: max_ks_score(a, b, seed=seed),
        "mean_psi": lambda a, b: mean_psi_score(a, b, seed=seed),
        "max_psi": lambda a, b: max_psi_score(a, b, seed=seed),
        "mmd_rbf": lambda a, b: mmd_rbf_score(a, b, seed=seed, max_samples=mmd_max_samples),
        "energy": lambda a, b: energy_distance_score(a, b, seed=seed, max_samples=energy_max_samples),
        "c2st": lambda a, b: classifier_two_sample_score(
            a,
            b,
            seed=seed,
            max_samples=c2st_max_samples,
            n_splits=c2st_n_splits,
            n_jobs=c2st_n_jobs,
        ),
        "sinkhorn": lambda a, b: sinkhorn_divergence_score(a, b, seed=seed),
    }


def compute_drift_scores(
    X_ref,
    X_cur,
    *,
    methods: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
    n_jobs: int = 1,
    swds_backend: str = "auto",
    swds_device: str | None = None,
    mmd_max_samples: int = 1000,
    energy_max_samples: int = 1000,
    c2st_max_samples: int = 4000,
    c2st_n_splits: int = 5,
    c2st_n_jobs: int | None = None,
) -> list[DriftScore]:
    config = DriftRuntimeConfig(
        seed=seed,
        n_jobs=n_jobs,
        swds_backend=swds_backend,
        swds_device=swds_device,
        mmd_max_samples=mmd_max_samples,
        energy_max_samples=energy_max_samples,
        c2st_max_samples=c2st_max_samples,
        c2st_n_splits=c2st_n_splits,
        c2st_n_jobs=c2st_n_jobs,
    )
    return DriftScorer(methods=methods, config=config).compute(X_ref, X_cur)


class DriftScorer:
    def __init__(
        self,
        *,
        methods: list[str] | tuple[str, ...] | None = None,
        config: DriftRuntimeConfig | None = None,
    ) -> None:
        self.methods = list(methods or DEFAULT_METHODS)
        self.config = config or DriftRuntimeConfig()
        self._registry = drift_method_registry(
            seed=self.config.seed,
            swds_backend=self.config.swds_backend,
            swds_device=self.config.swds_device,
            mmd_max_samples=self.config.mmd_max_samples,
            energy_max_samples=self.config.energy_max_samples,
            c2st_max_samples=self.config.c2st_max_samples,
            c2st_n_splits=self.config.c2st_n_splits,
            c2st_n_jobs=self.config.c2st_n_jobs,
        )
        self._prepared_swds: dict[str, object] = {}
        self._prepared_ref_id: int | None = None

    def prepare_reference(self, X_ref) -> None:
        self._prepared_swds = {}
        self._prepared_ref_id = id(X_ref)
        for method in self.methods:
            params = parse_swds_method(method)
            if params is None:
                continue
            n_projections, n_quantiles = params
            started = perf_counter()
            directions = random_directions(X_ref.shape[1], n_projections, seed=self.config.seed)
            self._prepared_swds[method] = prepare_sliced_wasserstein_reference(
                X_ref,
                directions=directions,
                n_quantiles=n_quantiles,
                backend=self.config.swds_backend,
                device=self.config.swds_device,
            )
            LOGGER.info(
                "prepared drift reference method=%s rows=%d runtime=%.3fs",
                method,
                X_ref.shape[0],
                perf_counter() - started,
            )

    def compute(self, X_ref, X_cur) -> list[DriftScore]:
        if self._prepared_ref_id is not None and self._prepared_ref_id != id(X_ref):
            LOGGER.debug("prepared reference invalidated because X_ref object changed")
            self._prepared_swds = {}
            self._prepared_ref_id = None
        LOGGER.debug(
            "computing drift scores methods=%s seed=%d ref_shape=%s cur_shape=%s n_jobs=%d",
            self.methods,
            self.config.seed,
            tuple(getattr(X_ref, "shape", ())),
            tuple(getattr(X_cur, "shape", ())),
            self.config.n_jobs,
        )
        if self.config.n_jobs == 1 or len(self.methods) <= 1:
            return [self._compute_one(method, X_ref, X_cur) for method in self.methods]
        scores = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
            delayed(self._compute_one)(method, X_ref, X_cur) for method in self.methods
        )
        return list(scores)

    def _compute_one(self, method: str, X_ref, X_cur) -> DriftScore:
        scorer = self._registry.get(method) or _dynamic_method(
            method,
            seed=self.config.seed,
            swds_backend=self.config.swds_backend,
            swds_device=self.config.swds_device,
        )
        if scorer is None:
            LOGGER.error("unknown drift method requested method=%s", method)
            raise ValueError(f"unknown drift method: {method!r}")
        LOGGER.debug("drift method started method=%s", method)
        started = perf_counter()
        if method in self._prepared_swds:
            score = float(score_sliced_wasserstein_prepared(self._prepared_swds[method], X_cur))
        else:
            score = float(scorer(X_ref, X_cur))
        elapsed = perf_counter() - started
        LOGGER.debug("drift method completed method=%s score=%.8f runtime=%.4fs", method, score, elapsed)
        return DriftScore(method=method, score=score, runtime_seconds=elapsed)


def parse_swds_method(method: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"swds(?:_k(?P<k>\d+))?(?:_q(?P<q>\d+))?", method)
    if not match:
        return None
    n_projections = int(match.group("k") or 128)
    n_quantiles = int(match.group("q") or 512)
    return n_projections, n_quantiles


def _dynamic_method(
    method: str,
    *,
    seed: int,
    swds_backend: str = "auto",
    swds_device: str | None = None,
) -> Callable | None:
    params = parse_swds_method(method)
    if params is None or method == "swds":
        return None
    n_projections, n_quantiles = params
    LOGGER.debug("using dynamic SWDS method method=%s n_projections=%d n_quantiles=%d", method, n_projections, n_quantiles)
    return lambda a, b: sliced_wasserstein_score(
        a,
        b,
        n_projections=n_projections,
        n_quantiles=n_quantiles,
        seed=seed,
        backend=swds_backend,
        device=swds_device,
    )
