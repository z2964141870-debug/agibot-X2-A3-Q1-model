"""Fixed training and holdout schedule for Q1 recovery experiments."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Condition:
    friction: float
    payload_kg: float = 0.
    delay_ticks: int = 0

TRAIN = {"low":Condition(.30), "nominal":Condition(.60), "high":Condition(.90)}
TEST = {"low":Condition(.42), "nominal":Condition(.60), "high":Condition(.78),
        "payload":Condition(.60,.8), "delay":Condition(.60,0.,1), "combined":Condition(.48,.8,1)}
TRAIN_IDS = (0, 4, 8, 12, 16, 17)
TRAIN_SEED, TEST_SEED = 2026112007, 2026112049
# Rank parameters without consulting holdout: mean and worst-surface terminal
# success first, then partial standing time to break sparse ties.
def utility(rows):
    import numpy as np
    rates=[np.mean([r["success"] for r in group]) for group in rows]
    partial=[np.mean([min(r["terminal_stable_ticks"],50)/50 for r in group]) for group in rows]
    height=[np.mean([min(r["end_z"],.39)/.39 for r in group]) for group in rows]
    return float(.65*np.mean(rates)+.25*min(rates)+.08*np.mean(partial)+.02*np.mean(height))

