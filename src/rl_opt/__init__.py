"""Synthetic ranking-policy world. Phase 0: the world and the Goodhart trap."""

from rl_opt.config import ACTIONS, DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.metrics import RolloutResult
from rl_opt.ranker import score_candidates
from rl_opt.trace import pick_canonical_user, trace_scenario
from rl_opt.world import TrueWorld

__all__ = [
    "ACTIONS",
    "DEFAULT_WEIGHTS",
    "ENGAGEMENT_WEIGHTS",
    "WorldParams",
    "RolloutResult",
    "score_candidates",
    "TrueWorld",
    "pick_canonical_user",
    "trace_scenario",
]
