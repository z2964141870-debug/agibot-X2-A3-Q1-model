"""RF16 has one independent short budget and no extension or recovery."""

from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"supermimic_q1_gait_comp"))
from overnight import OPTIMIZATION_DEADLINE, DELIVERY_DEADLINE, continuing_evidence

MAXIMUM_TRAINING_SECONDS=1300.
CHECKS=(300.,600.,1200.)

