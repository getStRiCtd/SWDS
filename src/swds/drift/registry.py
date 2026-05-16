from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from time import perf_counter
from typing import Callable

from joblib import Parallel, delayed
import numpy as np

from swds.drift.c2st import classifier_two_sample_score, classifier_two_sample_score_prepared, prepare_c2st_reference
from swds.drift.energy import energy_distance_score, energy_distance_score_prepared, prepare_energy_reference
from swds.drift.ks import ks_statistics, ks_statistics_prepared, max_ks_score, mean_ks_score, prepare_ks_reference
from swds.drift.mmd import mmd_rbf_score
from swds.drift.mmd import mmd_rbf_score_prepared, prepare_mmd_reference
from swds.drift.psi import max_psi_score, mean_psi_score, prepare_psi_reference, psi_statistics, psi_statistics_prepared
from swds.drift.sinkhorn import sinkhorn_divergence_score
from swds.drift.sliced_wasserstein import (
    prepare_sliced_wasserstein_reference,
    random_directions,
    score_sliced_wasserstein_prepared,
    sliced_wasserstein_score,
)


LOGGER = logging.getLogger(__name__)
_KS_METHODS = frozenset({"mean_ks", "max_ks"})
_PSI_METHODS = frozenset({"mean_psi", "max_psi"})


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
    ks_max_features: int | None = 2048
    psi_max_features: int | None = 2048
    psi_n_bins: int = 10
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
    ks_max_features: int | None = 2048,
    psi_max_features: int | None = 2048,
    psi_n_bins: int = 10,
    mmd_max_samples: int = 1000,
    energy_max_samples: int = 1000,
    c2st_max_samples: int = 4000,
    c2st_n_splits: int = 5,
    c2st_n_jobs: int | None = None,
) -> dict[str, Callable]:
    return {
        "swds": lambda a, b: sliced_wasserstein_score(a, b, seed=seed, backend=swds_backend, device=swds_device),
        "mean_ks": lambda a, b: mean_ks_score(a, b, max_features=ks_max_features, seed=seed),
        "max_ks": lambda a, b: max_ks_score(a, b, max_features=ks_max_features, seed=seed),
        "mean_psi": lambda a, b: mean_psi_score(a, b, n_bins=psi_n_bins, max_features=psi_max_features, seed=seed),
        "max_psi": lambda a, b: max_psi_score(a, b, n_bins=psi_n_bins, max_features=psi_max_features, seed=seed),
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
    ks_max_features: int | None = 2048,
    psi_max_features: int | None = 2048,
    psi_n_bins: int = 10,
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
        ks_max_features=ks_max_features,
        psi_max_features=psi_max_features,
        psi_n_bins=psi_n_bins,
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
            ks_max_features=self.config.ks_max_features,
            psi_max_features=self.config.psi_max_features,
            psi_n_bins=self.config.psi_n_bins,
            mmd_max_samples=self.config.mmd_max_samples,
            energy_max_samples=self.config.energy_max_samples,
            c2st_max_samples=self.config.c2st_max_samples,
            c2st_n_splits=self.config.c2st_n_splits,
            c2st_n_jobs=self.config.c2st_n_jobs,
        )
        self._prepared_swds: dict[str, object] = {}
        self._prepared_ks = None
        self._prepared_psi = None
        self._prepared_mmd = None
        self._prepared_energy = None
        self._prepared_c2st = None
        self._prepared_ref_id: int | None = None

    def prepare_reference(self, X_ref) -> None:
        self._clear_prepared()
        self._prepared_ref_id = id(X_ref)
        if any(method in _KS_METHODS for method in self.methods):
            started = perf_counter()
            self._prepared_ks = prepare_ks_reference(
                X_ref,
                max_features=self.config.ks_max_features,
                seed=self.config.seed,
            )
            LOGGER.info(
                "prepared drift reference family=ks rows=%d features=%d runtime=%.3fs",
                X_ref.shape[0],
                len(self._prepared_ks.feature_ids),
                perf_counter() - started,
            )
        if any(method in _PSI_METHODS for method in self.methods):
            started = perf_counter()
            self._prepared_psi = prepare_psi_reference(
                X_ref,
                n_bins=self.config.psi_n_bins,
                max_features=self.config.psi_max_features,
                seed=self.config.seed,
            )
            LOGGER.info(
                "prepared drift reference family=psi rows=%d features=%d bins=%d runtime=%.3fs",
                X_ref.shape[0],
                len(self._prepared_psi.features),
                self.config.psi_n_bins,
                perf_counter() - started,
            )
        if "mmd_rbf" in self.methods:
            started = perf_counter()
            self._prepared_mmd = prepare_mmd_reference(
                X_ref,
                max_samples=self.config.mmd_max_samples,
                seed=self.config.seed,
            )
            LOGGER.info(
                "prepared drift reference method=mmd_rbf rows=%d sampled_rows=%d runtime=%.3fs",
                X_ref.shape[0],
                self._prepared_mmd.x_ref.shape[0],
                perf_counter() - started,
            )
        if "energy" in self.methods:
            started = perf_counter()
            self._prepared_energy = prepare_energy_reference(
                X_ref,
                max_samples=self.config.energy_max_samples,
                seed=self.config.seed,
            )
            LOGGER.info(
                "prepared drift reference method=energy rows=%d sampled_rows=%d runtime=%.3fs",
                X_ref.shape[0],
                self._prepared_energy.x_ref.shape[0],
                perf_counter() - started,
            )
        if "c2st" in self.methods:
            started = perf_counter()
            self._prepared_c2st = prepare_c2st_reference(
                X_ref,
                max_samples=self.config.c2st_max_samples,
                seed=self.config.seed,
            )
            LOGGER.info(
                "prepared drift reference method=c2st rows=%d sampled_rows=%d runtime=%.3fs",
                X_ref.shape[0],
                self._prepared_c2st.X_ref_sub.shape[0],
                perf_counter() - started,
            )
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
            self._clear_prepared()
        LOGGER.debug(
            "computing drift scores methods=%s seed=%d ref_shape=%s cur_shape=%s n_jobs=%d",
            self.methods,
            self.config.seed,
            tuple(getattr(X_ref, "shape", ())),
            tuple(getattr(X_cur, "shape", ())),
            self.config.n_jobs,
        )
        tasks = self._compute_tasks(X_ref, X_cur)
        if self.config.n_jobs == 1 or len(tasks) <= 1:
            chunks = [compute() for _, compute in tasks]
        else:
            chunks = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(compute)() for _, compute in tasks
            )
        by_method: dict[str, DriftScore] = {}
        for chunk in chunks:
            for score in chunk:
                by_method[score.method] = score
        return [by_method[method] for method in self.methods]

    def _clear_prepared(self) -> None:
        self._prepared_swds = {}
        self._prepared_ks = None
        self._prepared_psi = None
        self._prepared_mmd = None
        self._prepared_energy = None
        self._prepared_c2st = None
        self._prepared_ref_id = None

    def _compute_tasks(self, X_ref, X_cur) -> list[tuple[tuple[str, ...], Callable[[], list[DriftScore]]]]:
        tasks: list[tuple[tuple[str, ...], Callable[[], list[DriftScore]]]] = []
        consumed: set[str] = set()
        ks_methods = tuple(dict.fromkeys(method for method in self.methods if method in _KS_METHODS))
        if ks_methods:
            tasks.append((ks_methods, lambda methods=ks_methods: self._compute_ks_group(methods, X_ref, X_cur)))
            consumed.update(_KS_METHODS)
        psi_methods = tuple(dict.fromkeys(method for method in self.methods if method in _PSI_METHODS))
        if psi_methods:
            tasks.append((psi_methods, lambda methods=psi_methods: self._compute_psi_group(methods, X_ref, X_cur)))
            consumed.update(_PSI_METHODS)

        seen = set(consumed)
        for method in self.methods:
            if method in seen:
                continue
            seen.add(method)
            tasks.append(((method,), lambda method=method: [self._compute_one(method, X_ref, X_cur)]))
        return tasks

    def _compute_ks_group(self, methods: tuple[str, ...], X_ref, X_cur) -> list[DriftScore]:
        LOGGER.debug("drift method group started family=ks methods=%s", methods)
        started = perf_counter()
        if self._prepared_ks is not None:
            stats = ks_statistics_prepared(self._prepared_ks, X_cur)
        else:
            stats = ks_statistics(X_ref, X_cur, max_features=self.config.ks_max_features, seed=self.config.seed)
        elapsed = perf_counter() - started
        return self._scores_from_statistics(methods, stats, elapsed)

    def _compute_psi_group(self, methods: tuple[str, ...], X_ref, X_cur) -> list[DriftScore]:
        LOGGER.debug("drift method group started family=psi methods=%s", methods)
        started = perf_counter()
        if self._prepared_psi is not None:
            stats = psi_statistics_prepared(self._prepared_psi, X_cur)
        else:
            stats = psi_statistics(
                X_ref,
                X_cur,
                n_bins=self.config.psi_n_bins,
                max_features=self.config.psi_max_features,
                seed=self.config.seed,
            )
        elapsed = perf_counter() - started
        return self._scores_from_statistics(methods, stats, elapsed)

    def _scores_from_statistics(self, methods: tuple[str, ...], stats, elapsed: float) -> list[DriftScore]:
        runtime = elapsed / max(len(methods), 1)
        scores = []
        for method in methods:
            if len(stats) == 0:
                value = float("nan")
            elif method.startswith("mean_"):
                value = float(np.nanmean(stats))
            elif method.startswith("max_"):
                value = float(np.nanmax(stats))
            else:
                raise ValueError(f"unknown grouped drift method: {method!r}")
            LOGGER.debug(
                "drift grouped method completed method=%s score=%.8f runtime_share=%.4fs group_runtime=%.4fs",
                method,
                value,
                runtime,
                elapsed,
            )
            scores.append(DriftScore(method=method, score=value, runtime_seconds=runtime))
        return scores

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
        elif method == "mmd_rbf" and self._prepared_mmd is not None:
            score = float(
                mmd_rbf_score_prepared(
                    self._prepared_mmd,
                    X_cur,
                    max_samples=self.config.mmd_max_samples,
                    seed=self.config.seed,
                )
            )
        elif method == "energy" and self._prepared_energy is not None:
            score = float(
                energy_distance_score_prepared(
                    self._prepared_energy,
                    X_cur,
                    max_samples=self.config.energy_max_samples,
                    seed=self.config.seed,
                )
            )
        elif method == "c2st" and self._prepared_c2st is not None:
            score = float(
                classifier_two_sample_score_prepared(
                    self._prepared_c2st,
                    X_cur,
                    max_samples=self.config.c2st_max_samples,
                    seed=self.config.seed,
                    n_splits=self.config.c2st_n_splits,
                    n_jobs=self.config.c2st_n_jobs,
                )
            )
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
