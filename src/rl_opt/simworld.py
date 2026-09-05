"""Fitted surrogate: theta + predicted bait share → delayed V.

Not a copy of the 2-day proxy. Training labels are full-horizon retention
from charged `Harness.label` calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_opt.harness import Receipt
from rl_opt.space import LOG_HI, LOG_LO, to_theta


def _scale_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    return (np.asarray(theta, dtype=np.float64) - LOG_LO) / (LOG_HI - LOG_LO)


def _ridge(X: NDArray[np.float64], y: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    d = X.shape[1]
    gram = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(gram, X.T @ y)


@dataclass
class SimWorld:
    bait_coef: NDArray[np.float64]
    v_coef: NDArray[np.float64]
    n_train: int

    @classmethod
    def fit(cls, rows: list[Receipt], lam: float = 0.5) -> "SimWorld":
        labeled = [r for r in rows if r.value is not None]
        if len(labeled) < 3:
            raise ValueError("need at least 3 full-horizon labels to fit SimWorld")
        thetas = np.stack([_scale_theta(to_theta(np.array(r.weights))) for r in labeled])
        bait = np.array([r.bait_share for r in labeled])
        v = np.array([r.value for r in labeled])
        n = len(labeled)
        ones = np.ones((n, 1))
        x_bait = np.hstack([ones, thetas])
        x_v = np.hstack([ones, thetas, bait[:, None]])
        return cls(bait_coef=_ridge(x_bait, bait, lam), v_coef=_ridge(x_v, v, lam), n_train=n)

    def predict_bait(self, weights: NDArray[np.float64]) -> float:
        z = _scale_theta(to_theta(weights))
        x = np.concatenate([[1.0], z])
        return float(np.clip(x @ self.bait_coef, 0.0, 1.0))

    def predict_v(self, weights: NDArray[np.float64]) -> float:
        z = _scale_theta(to_theta(weights))
        bait = self.predict_bait(weights)
        x = np.concatenate([[1.0], z, [bait]])
        return float(np.clip(x @ self.v_coef, 0.0, 1.0))

    def rollout(self, weights: NDArray[np.float64]) -> float:
        """Free query. Same name as TrueWorld so M1 does not special-case."""
        return self.predict_v(weights)


@dataclass
class GaussianProcessSimWorld:
    """Small-data RBF surrogate fitted directly to delayed V."""

    x_train: NDArray[np.float64]
    alpha: NDArray[np.float64]
    chol: NDArray[np.float64]
    y_mean: float
    y_scale: float
    length_scale: float
    noise: float

    @staticmethod
    def _kernel(
        a: NDArray[np.float64], b: NDArray[np.float64], length_scale: float
    ) -> NDArray[np.float64]:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-0.5 * d2 / length_scale**2)

    @classmethod
    def fit(cls, rows: list[Receipt]) -> "GaussianProcessSimWorld":
        labeled = [row for row in rows if row.value is not None]
        if len(labeled) < 3:
            raise ValueError("need at least 3 full-horizon labels to fit GP")
        x = np.stack(
            [_scale_theta(to_theta(np.asarray(row.weights))) for row in labeled]
        )
        y = np.asarray([row.value for row in labeled], dtype=np.float64)
        y_mean = float(y.mean())
        y_scale = max(float(y.std()), 0.01)
        yn = (y - y_mean) / y_scale

        # Hyperparameters are selected from the labels themselves by leave-one-
        # out error, so no evaluation seed is used to tune the surrogate.
        choices = [
            (length_scale, noise)
            for length_scale in (0.2, 0.35, 0.5, 0.8, 1.2)
            for noise in (0.03, 0.08, 0.15)
        ]
        best = choices[0]
        best_error = np.inf
        eye = np.eye(len(y))
        for length_scale, noise in choices:
            k = cls._kernel(x, x, length_scale) + noise**2 * eye
            try:
                k_inv = np.linalg.inv(k)
            except np.linalg.LinAlgError:
                continue
            alpha = k_inv @ yn
            loo_residual = alpha / np.clip(np.diag(k_inv), 1e-9, None)
            error = float(np.mean(loo_residual**2))
            if error < best_error:
                best_error = error
                best = (length_scale, noise)

        length_scale, noise = best
        k = cls._kernel(x, x, length_scale) + noise**2 * eye
        chol = np.linalg.cholesky(k + 1e-8 * eye)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yn))
        return cls(x, alpha, chol, y_mean, y_scale, length_scale, noise)

    def predict(
        self, weights: NDArray[np.float64]
    ) -> tuple[float, float]:
        x = _scale_theta(to_theta(weights))[None, :]
        k_star = self._kernel(self.x_train, x, self.length_scale)[:, 0]
        mean = self.y_mean + self.y_scale * float(k_star @ self.alpha)
        solved = np.linalg.solve(self.chol, k_star)
        variance = max(1e-9, 1.0 - float(solved @ solved))
        std = self.y_scale * np.sqrt(variance)
        return float(np.clip(mean, 0.0, 1.0)), float(std)

    def predict_v(self, weights: NDArray[np.float64]) -> float:
        return self.predict(weights)[0]
