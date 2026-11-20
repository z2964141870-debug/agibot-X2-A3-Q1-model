"""Fixed schedules for command-delay training, selection and independent tests."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# (sliding friction, pelvis-centred point load kg, transport mode)
SURFACES={"normal":(.60,0.,"now"),"lag":(.60,0.,"all"),"rough":(.72,0.,"all"),
          "slick":(.47,0.,"all"),"load":(.60,.7,"all"),"mixed":(.51,.7,"all"),
          "recovery_only":(.60,0.,"recover"),"outside_recovery":(.60,0.,"outside")}
SCHEDULE={"collect0":(230011,24,5),"collect1":(230021,24,5),"valid":(230051,24,5),
          "tune":(230073,9,5),"test0":(230111,24,10),"test1":(230137,24,10),
          "audit":(2026112071,24,5)}
EVALUATION=("normal","lag","slick","load","mixed")
GATE=("Preserve all nominal outcomes by bypass; improve 20 ms static and dynamic counts separately; "
      "no increase in issued estimated effort violations; report all individual scenes.")

