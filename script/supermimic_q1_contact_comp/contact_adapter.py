"""Independent phase-aware adapter; immutable gait adapter implementation reused."""

from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"supermimic_q1_gait_comp"))
from gait_adapter import Adapter as OriginalAdapter, Actor, make_ppo, bootstrap_rewards, termination_masks
from contact_reference import ContactReference


class Adapter(OriginalAdapter):
    def __init__(self,env,teacher,thresholds,upper_alpha=.5):
        super().__init__(env,teacher,thresholds,upper_alpha)
        self.foot_reference=ContactReference(env.supermimic_mdp)
        self.observations,self.nominal,self.reference_action=self.build(
            self.history,self.current,self.gate.alpha,self.gate.filtered,
            self.gate.cooldown,self.caps())

