# Random Forest surrogate model that screens candidate sampling sets before
# they are evaluated with lightweight fine-tuning (Section 3.2).
#
# Each candidate is summarized by statistical features of its coordinate set
# (band shares, distance statistics, occupancy entropy per module — see
# space.solution_features). The surrogate is a RandomForestRegressor trained
# online on all (features, heuristic intensity) pairs observed so far. During
# the search, only candidates whose predicted intensity falls in the top
# quantile are evaluated with real fine-tuning, which removes the majority of
# the evaluation cost (Section 4.4).

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class RFSurrogate:
    def __init__(self, n_estimators: int = 100, min_samples: int = 10,
                 seed: int = 0):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        self.min_samples = min_samples
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def observe(self, features: np.ndarray, intensity: float) -> None:
        """Record a (features, intensity) pair for online training."""
        self.X.append(np.asarray(features, dtype=np.float64))
        self.y.append(float(intensity))

    @property
    def ready(self) -> bool:
        return len(self.y) >= self.min_samples

    def fit(self) -> Optional[float]:
        """Refit on all observations; returns out-of-bag style R^2 on
        the training data or None when not enough samples exist."""
        if not self.ready:
            return None
        X = np.stack(self.X)
        y = np.array(self.y)
        self.model.fit(X, y)
        return float(self.model.score(X, y))

    def predict(self, features_list: List[np.ndarray]) -> np.ndarray:
        """Predict the heuristic intensity for a batch of candidates."""
        X = np.stack(features_list)
        return self.model.predict(X)

    def select_top(self, features_list: List[np.ndarray],
                   quantile: float = 0.3) -> List[int]:
        """Return indices of the candidates whose predicted intensity is in
        the top `quantile` fraction (default: evaluate the best 30%)."""
        if not self.ready or len(features_list) == 0:
            return list(range(len(features_list)))
        preds = self.predict(features_list)
        k = max(1, int(len(preds) * quantile))
        return list(np.argsort(preds)[-k:])
