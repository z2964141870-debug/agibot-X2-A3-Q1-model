from protocol import Condition

LATE_TRAIN = {"low":Condition(.30),"midlow":Condition(.46),"nominal":Condition(.60),
              "midhigh":Condition(.74),"high":Condition(.90)}
LATE_TEST = {"low":Condition(.39),"nominal":Condition(.60),"high":Condition(.81),
             "payload":Condition(.60,1.),"delay":Condition(.60,0.,1),"combined":Condition(.51,1.,1)}
LATE_SEED=2026112071

