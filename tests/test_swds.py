import unittest

import numpy as np

from swds.drift.registry import DriftRuntimeConfig, DriftScorer, compute_drift_scores
from swds.drift.sliced_wasserstein import sliced_wasserstein_score


class SlicedWassersteinTests(unittest.TestCase):
    def test_mean_shift_increases_score(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(size=(500, 6))
        near = rng.normal(size=(500, 6))
        shifted = rng.normal(loc=1.25, size=(500, 6))

        near_score = sliced_wasserstein_score(ref, near, n_projections=64, n_quantiles=128, seed=1)
        shifted_score = sliced_wasserstein_score(ref, shifted, n_projections=64, n_quantiles=128, seed=1)

        self.assertGreater(shifted_score, near_score * 2.0)

    def test_subsample_mode_runs(self):
        rng = np.random.default_rng(9)
        ref = rng.normal(size=(300, 4))
        cur = rng.normal(size=(180, 4))
        score = sliced_wasserstein_score(
            ref,
            cur,
            n_projections=16,
            mode="subsample",
            subsample_size=100,
            n_subsample_seeds=2,
            seed=2,
        )
        self.assertTrue(np.isfinite(score))

    def test_dynamic_projection_count_method_runs(self):
        rng = np.random.default_rng(10)
        ref = rng.normal(size=(100, 5))
        cur = rng.normal(size=(100, 5))
        scores = compute_drift_scores(ref, cur, methods=("swds_k8", "swds_k16_q64"), seed=4)
        self.assertEqual([score.method for score in scores], ["swds_k8", "swds_k16_q64"])
        self.assertTrue(all(np.isfinite(score.score) for score in scores))

    def test_prepared_reference_matches_direct_swds(self):
        rng = np.random.default_rng(11)
        ref = rng.normal(size=(120, 5))
        cur = rng.normal(size=(80, 5))
        direct = compute_drift_scores(ref, cur, methods=("swds_k8_q64",), seed=5)[0].score
        scorer = DriftScorer(methods=("swds_k8_q64",), config=DriftRuntimeConfig(seed=5, swds_backend="numpy"))
        scorer.prepare_reference(ref)
        prepared = scorer.compute(ref, cur)[0].score
        self.assertAlmostEqual(direct, prepared, places=10)


if __name__ == "__main__":
    unittest.main()
