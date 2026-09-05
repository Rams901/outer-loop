"""Ground-truth generative world. Policy-dependent by construction.

The ranker never sees satisfaction. It sees predicted action probabilities, and
bait content is genuinely engaging on like/dwell — that is the trap, not a
prediction bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_opt.config import ACTION_INDEX, N_ACTIONS, WorldParams
from rl_opt.metrics import RolloutResult, from_counters
from rl_opt.ranker import rank_indices

# Intercepts chosen so median-user, median-content base rates span ~3 orders.
# like ~ 8e-2, reply ~ 8e-3, dwell ~ 1.8e-1, share ~ 4e-3, hide ~ 6e-3, report ~ 8e-5
BASE_LOGIT = np.array([-2.44, -4.82, -1.52, -5.52, -5.11, -9.43], dtype=np.float64)

# Columns match ACTIONS. Bait lifts cheap engagement and barely touches report/hide.
COEF_REL = np.array([1.20, 1.40, 1.00, 1.00, -1.20, -0.40], dtype=np.float64)
COEF_QUALITY = np.array([0.55, 1.90, 0.25, 2.30, -1.10, -0.90], dtype=np.float64)
COEF_BAIT = np.array([0.90, -0.40, 1.05, -0.55, 0.06, 0.02], dtype=np.float64)
COEF_TOX = np.array([-0.35, -1.20, -0.25, -1.80, 2.60, 3.80], dtype=np.float64)

SAT_QUALITY = 0.75
SAT_RELEVANCE = 0.20
SAT_BAIT = 1.20
SAT_TOX = 0.70
# Small enough that 21 days do not clip satisfaction against 0.98; that clip
# was flattening V so default looked optimal.
SAT_STEP = 0.032
SAT_INIT = 0.48
SAT_DECAY_INACTIVE = 0.97
RETURN_K = 8.0
RETURN_CENTER = 0.48


def item_utility(
    quality: NDArray[np.float64],
    bait: NDArray[np.float64],
    toxicity: NDArray[np.float64],
    relevance: NDArray[np.float64],
) -> NDArray[np.float64]:
    return SAT_QUALITY * quality + SAT_RELEVANCE * relevance - SAT_BAIT * bait - SAT_TOX * toxicity


def satisfaction_delta(sensitivity: NDArray[np.float64] | float, item_mean: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return SAT_STEP * sensitivity * item_mean


def p_return(satisfaction: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    x = RETURN_K * (np.asarray(satisfaction, dtype=np.float64) - RETURN_CENTER)
    return _sigmoid(x)


def sample_candidates(
    rng: np.random.Generator, n_users: int, n_posts: int, n_cand: int
) -> NDArray[np.int64]:
    scores = rng.random((n_users, n_posts))
    return np.argpartition(scores, n_cand, axis=1)[:, :n_cand].astype(np.int64)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class Catalogue:
    quality: NDArray[np.float64]
    bait: NDArray[np.float64]
    toxicity: NDArray[np.float64]
    topics: NDArray[np.float64]
    cold_start: NDArray[np.bool_]
    pred_error: NDArray[np.float64]


@dataclass
class Population:
    interests: NDArray[np.float64]
    propensity: NDArray[np.float64]
    sensitivity: NDArray[np.float64]


class TrueWorld:
    def __init__(self, params: WorldParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.last_pixel_log: dict | None = None
        rng = np.random.default_rng(seed)
        self.population = self._make_population(rng)
        self.catalogue = self._make_catalogue(rng)
        self._true_logits = self._precompute_true_logits()
        self._pred_probs = _sigmoid(self._true_logits + self.catalogue.pred_error[None, :, :])

    @classmethod
    def generate(cls, params: WorldParams, seed: int) -> "TrueWorld":
        return cls(params, seed)

    def _make_population(self, rng: np.random.Generator) -> Population:
        p = self.params
        interests = rng.dirichlet(np.full(p.n_topics, p.interest_concentration), size=p.n_users)
        propensity = rng.normal(0.0, p.propensity_spread, size=(p.n_users, N_ACTIONS))
        sensitivity = rng.uniform(0.6, 1.4, size=p.n_users)
        return Population(interests=interests, propensity=propensity, sensitivity=sensitivity)

    def _make_catalogue(self, rng: np.random.Generator) -> Catalogue:
        p = self.params
        n = p.n_posts
        is_bait = rng.random(n) < p.bait_fraction
        quality = np.where(is_bait, rng.beta(1.6, 4.5, size=n), rng.beta(4.5, 1.8, size=n))
        bait = np.where(is_bait, rng.uniform(0.65, 1.0, size=n), rng.beta(1.2, 8.0, size=n))
        bait = bait * p.bait_detectability
        toxicity = rng.beta(1.1, 14.0, size=n)
        topics = rng.dirichlet(np.full(p.n_topics, 0.8), size=n)
        cold = rng.random(n) < p.cold_start_fraction
        noise_scale = np.where(cold, p.cold_start_noise, 1.0).astype(np.float64)
        # Rare actions are worse-calibrated. Heteroskedastic in the log-odds.
        action_scale = np.array([0.6, 0.9, 0.6, 1.0, 1.2, 1.6], dtype=np.float64)
        pred_error = rng.normal(0.0, 1.0, size=(n, N_ACTIONS))
        pred_error *= (p.pred_noise * noise_scale)[:, None] * action_scale[None, :]
        return Catalogue(
            quality=quality,
            bait=bait,
            toxicity=toxicity,
            topics=topics,
            cold_start=cold,
            pred_error=pred_error,
        )

    def _precompute_true_logits(self) -> NDArray[np.float64]:
        """(n_users, n_posts, n_actions) logits, no position bias."""
        rel = self.population.interests @ self.catalogue.topics.T
        q = self.catalogue.quality
        b = self.catalogue.bait
        t = self.catalogue.toxicity
        logits = (
            BASE_LOGIT[None, None, :]
            + self.population.propensity[:, None, :]
            + COEF_REL[None, None, :] * rel[:, :, None]
            + COEF_QUALITY[None, None, :] * q[None, :, None]
            + COEF_BAIT[None, None, :] * b[None, :, None]
            + COEF_TOX[None, None, :] * t[None, :, None]
        )
        return logits

    def _clone_shell(self) -> "TrueWorld":
        other = TrueWorld.__new__(TrueWorld)
        other.params = self.params
        other.seed = self.seed
        other.last_pixel_log = None
        other.population = self.population
        other.catalogue = self.catalogue
        other._true_logits = self._true_logits
        other._pred_probs = self._pred_probs
        return other

    def with_rotated_interests(self, shift: int = 1) -> "TrueWorld":
        """Same catalogue, rotated topic tastes — a stale population (Q8)."""
        other = self._clone_shell()
        interests = np.roll(self.population.interests, int(shift) % self.params.n_topics, axis=1)
        other.population = Population(
            interests=interests,
            propensity=self.population.propensity,
            sensitivity=self.population.sensitivity,
        )
        other._true_logits = other._precompute_true_logits()
        other._pred_probs = _sigmoid(other._true_logits + other.catalogue.pred_error[None, :, :])
        return other

    def with_rare_pred_scale(self, scale: float) -> "TrueWorld":
        """Miscalibrated hide/report predictions (Q8 rare-rate axis)."""
        other = self._clone_shell()
        pred_error = self.catalogue.pred_error.copy()
        pred_error[:, ACTION_INDEX["hide"]] *= float(scale)
        pred_error[:, ACTION_INDEX["report"]] *= float(scale)
        other.catalogue = Catalogue(
            quality=self.catalogue.quality,
            bait=self.catalogue.bait,
            toxicity=self.catalogue.toxicity,
            topics=self.catalogue.topics,
            cold_start=self.catalogue.cold_start,
            pred_error=pred_error,
        )
        other._true_logits = self._true_logits
        other._pred_probs = _sigmoid(other._true_logits + pred_error[None, :, :])
        return other

    def primary_topic(self) -> NDArray[np.int64]:
        return np.argmax(self.population.interests, axis=1).astype(np.int64)

    def relevance(self) -> NDArray[np.float64]:
        return self.population.interests @ self.catalogue.topics.T

    def true_probs(self) -> NDArray[np.float64]:
        return _sigmoid(self._true_logits)

    def predicted_probs(self) -> NDArray[np.float64]:
        return self._pred_probs

    def action_base_rates(self) -> dict[str, float]:
        """Mean true probability across users and posts, no position."""
        means = self.true_probs().mean(axis=(0, 1))
        from rl_opt.config import ACTIONS

        return {name: float(means[i]) for i, name in enumerate(ACTIONS)}

    def bait_vs_quality_likes(self) -> tuple[float, float]:
        p = self.true_probs()[:, :, ACTION_INDEX["like"]]
        bait = self.catalogue.bait > 0.5
        return float(p[:, bait].mean()), float(p[:, ~bait].mean())

    def rollout(
        self,
        weights: NDArray[np.float64],
        n_days: int | None = None,
        seed: int = 0,
        user_ids: NDArray[np.int64] | None = None,
        watch_ids: NDArray[np.int64] | None = None,
    ) -> RolloutResult:
        p = self.params
        days = p.horizon_days if n_days is None else int(n_days)
        if days < 1:
            raise ValueError("n_days must be >= 1")
        if user_ids is None:
            index = np.arange(p.n_users, dtype=np.int64)
            rng = np.random.default_rng(self._rollout_seed(seed))
        else:
            index = np.asarray(user_ids, dtype=np.int64)
            rng = np.random.default_rng(
                self._rollout_seed(seed) ^ (int(index[0]) * 1_000_003 + len(index))
            )
        n_users = int(index.size)
        satisfaction = np.full(n_users, SAT_INIT, dtype=np.float64)
        pred = self._pred_probs[index]
        true_p = _sigmoid(self._true_logits[index])
        rel = self.relevance()[index]
        cat = self.catalogue
        sensitivity = self.population.sensitivity[index]
        topics = np.argmax(self.population.interests[index], axis=1).astype(np.int64)

        watch_pairs: list[tuple[int, int]] = []
        if watch_ids is not None:
            loc_of = {int(uid): i for i, uid in enumerate(index)}
            watch_pairs = [(int(u), loc_of[int(u)]) for u in np.asarray(watch_ids).tolist() if int(u) in loc_of]
        pixel_days: list[list[dict]] = [[] for _ in range(days)] if watch_pairs else []

        action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
        impressions = 0
        bait_shown = 0
        proxy_engagement = 0
        proxy_impressions = 0
        returned = np.zeros((days, n_users), dtype=np.int8)
        returned[0] = 1

        daily_return = np.zeros(days, dtype=np.float64)
        daily_bait = np.zeros(days, dtype=np.float64)
        daily_engagement = np.zeros(days, dtype=np.float64)
        daily_satisfaction = np.zeros(days, dtype=np.float64)
        like_i = ACTION_INDEX["like"]
        dwell_i = ACTION_INDEX["dwell"]
        hide_i = ACTION_INDEX["hide"]
        slots = np.arange(p.feed_size, dtype=np.float64)
        pos_mult = p.position_decay**slots
        rel_cut = float(np.median(rel)) if rel.size else 0.5

        for day in range(days):
            if day > 0:
                returned[day] = rng.random(n_users) < p_return(satisfaction)
            active = np.flatnonzero(returned[day] == 1)
            shown = np.zeros((0, p.feed_size), dtype=np.int64)
            draws = np.zeros((0, p.feed_size, N_ACTIONS), dtype=bool)
            bait_items = np.zeros((0, p.feed_size), dtype=np.float64)
            item = np.zeros((0, p.feed_size), dtype=np.float64)
            r_feed = np.zeros((0, p.feed_size), dtype=np.float64)
            if active.size:
                cand = sample_candidates(rng, active.size, p.n_posts, p.n_candidates)
                pred_c = pred[active[:, None], cand]
                chosen_local = rank_indices(pred_c, weights, p.feed_size)
                shown = np.take_along_axis(cand, chosen_local, axis=1)

                true_c = true_p[active[:, None], shown]
                shown_p = np.clip(true_c * pos_mult[None, :, None], 1e-12, 1.0 - 1e-12)
                draws = rng.random(shown_p.shape) < shown_p

                n_imp = int(active.size * p.feed_size)
                impressions += n_imp
                action_counts += draws.reshape(-1, N_ACTIONS).sum(axis=0)
                bait_items = cat.bait[shown]
                bait_shown += int((bait_items > 0.5).sum())

                if day < p.proxy_days:
                    proxy_impressions += n_imp
                    proxy_engagement += int(draws[:, :, like_i].sum() + draws[:, :, dwell_i].sum())

                q = cat.quality[shown]
                b = cat.bait[shown]
                t = cat.toxicity[shown]
                r_feed = np.take_along_axis(rel[active], shown, axis=1)
                item = item_utility(q, b, t, r_feed)
                delta = satisfaction_delta(sensitivity[active], item.mean(axis=1))
                satisfaction[active] = np.clip(satisfaction[active] + delta, 0.02, 0.98)
            inactive = returned[day] == 0
            if day > 0 and inactive.any():
                satisfaction[inactive] = np.clip(satisfaction[inactive] * SAT_DECAY_INACTIVE, 0.02, 0.98)

            n_imp_day = int(active.size * p.feed_size)
            daily_return[day] = float(returned[day].mean())
            daily_bait[day] = float((bait_items > 0.5).mean()) if n_imp_day else 0.0
            daily_engagement[day] = float(
                (draws[:, :, like_i].sum() + draws[:, :, dwell_i].sum()) / max(n_imp_day, 1)
            )
            daily_satisfaction[day] = float(satisfaction.mean())

            if watch_pairs:
                row_of = {int(a): i for i, a in enumerate(active.tolist())}
                for uid, loc in watch_pairs:
                    rec: dict = {
                        "user_id": uid,
                        "returned": bool(returned[day, loc]),
                        "satisfaction": float(satisfaction[loc]),
                        "topic": int(topics[loc]),
                        "feed": [],
                        "bait_share": 0.0,
                        "engagement": 0.0,
                        "n_likes": 0,
                        "n_dwells": 0,
                        "n_hides": 0,
                        "n_in_interest": 0,
                        "n_oon": 0,
                        "n_bait": 0,
                        "delta_s": 0.0,
                    }
                    row = row_of.get(loc)
                    if rec["returned"] and row is not None:
                        posts = shown[row]
                        likes = draws[row, :, like_i]
                        dwells = draws[row, :, dwell_i]
                        hides = draws[row, :, hide_i]
                        bait_f = bait_items[row] > 0.5
                        rel_f = r_feed[row]
                        in_int = rel_f >= rel_cut
                        rec["n_likes"] = int(likes.sum())
                        rec["n_dwells"] = int(dwells.sum())
                        rec["n_hides"] = int(hides.sum())
                        rec["n_bait"] = int(bait_f.sum())
                        rec["n_in_interest"] = int(in_int.sum())
                        rec["n_oon"] = int((~in_int).sum())
                        rec["bait_share"] = float(bait_f.mean())
                        rec["engagement"] = float((likes.sum() + dwells.sum()) / max(p.feed_size, 1))
                        rec["delta_s"] = float(item[row].mean()) if item.shape[0] else 0.0
                        rec["feed"] = [
                            {
                                "post_id": int(posts[s]),
                                "slot": s,
                                "bait": bool(bait_f[s]),
                                "quality": float(cat.quality[posts[s]]),
                                "relevance": float(rel_f[s]),
                                "in_interest": bool(in_int[s]),
                                "like": bool(likes[s]),
                                "dwell": bool(dwells[s]),
                                "hide": bool(hides[s]),
                            }
                            for s in range(p.feed_size)
                        ]
                    pixel_days[day].append(rec)

        result = from_counters(
            impressions=impressions,
            actions=action_counts,
            bait_shown=bait_shown,
            returned=returned,
            satisfaction=satisfaction,
            proxy_engagement=proxy_engagement,
            proxy_impressions=proxy_impressions,
            params=p,
            daily_return=daily_return,
            daily_bait_share=daily_bait,
            daily_engagement=daily_engagement,
            daily_satisfaction=daily_satisfaction,
        )
        if watch_pairs:
            self.last_pixel_log = {
                "schema": "rl_opt.pixel.v1",
                "world_seed": self.seed,
                "rollout_seed": seed,
                "weights": np.asarray(weights, dtype=np.float64).tolist(),
                "rel_cut": rel_cut,
                "watch_ids": [u for u, _ in watch_pairs],
                "users": [
                    {
                        "user_id": uid,
                        "topic": int(topics[loc]),
                        "sensitivity": float(sensitivity[loc]),
                    }
                    for uid, loc in watch_pairs
                ],
                "cohort": {
                    "daily_return": daily_return.tolist(),
                    "daily_bait_share": daily_bait.tolist(),
                    "daily_engagement": daily_engagement.tolist(),
                    "daily_satisfaction": daily_satisfaction.tolist(),
                    "value": result.value,
                    "proxy": result.proxy,
                },
                "days": pixel_days,
            }
        else:
            self.last_pixel_log = None
        return result

    def _rollout_seed(self, seed: int) -> int:
        # Paired comparisons share the starting RNG. Rankings still diverge it.
        return int((self.seed * 1_000_003 + seed) % (2**31 - 1))
