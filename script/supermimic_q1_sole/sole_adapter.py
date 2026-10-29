"""Use an explicit independent sole reward without changing RF16 action or PPO APIs."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_contact_comp"))
from contact_adapter import Adapter as ContactAdapter
from sole_reference import SoleReference


class Adapter(ContactAdapter):
    def __init__(self, env, teacher, thresholds, upper_alpha=.5, *, scene):
        super().__init__(env, teacher, thresholds, upper_alpha)
        self.foot_reference = SoleReference(env.supermimic_mdp, scene)
