"""E52 Exp-1a three-arm Adam-projection finetune constants; no tensor imports."""

UPDATES = 8
OPTIMIZER_STEPS = UPDATES * 4
TRANSITIONS = UPDATES * 16 * 24
CHECKPOINTS = (2, 4, 8)
ARMS = ("A", "B", "C")
SOURCE_EXPERIMENT = "A3_ZHY_CLOTH_FAILURE_20261013_E45"
SOURCE_RUN = "data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0"
SOURCE_CHECKPOINT = SOURCE_RUN + "/milestone_4/model_step_000004.pt"
SOURCE_UPDATES = 4
SOURCE_OPTIMIZER_STEPS = 16

BASE_LR = 2e-6
LR_SCALE_ARM_C = 0.1
ARMS_LAUNCH_OVERRIDES = {
    "A": tuple(),
    "B": tuple(),
    "C": ("algo.config.actor_learning_rate=2e-7",
          "algo.config.adaptive_lr_min=1e-7",
          "algo.config.adaptive_lr_max=2e-6"),
}
ARMS_TRAINER = {
    "A": "e52_trainer.E52ArmATrainer",
    "B": "e52_trainer.E52ResumeTrainer",
    "C": "e52_trainer.E52ResumeTrainer",
}
ARMS_NOTES = {
    "A": "original Adam displacement + original teacher 0.05 hard gate (gate intercepts; gate-hit step is a reading)",
    "B": "each Adam displacement projected with the E50 teacher first-order non-increasing half-space operator before commit; teacher gate records without intercepting",
    "C": "original Adam displacement at LR x0.1 (base and adaptive bounds all x0.1); teacher gate records without intercepting",
}

STAGES = {
    "prepare": ("REGISTERED", "PREPARED"),
    "train-arm-a": ("PREPARED", "ARM_A_TRAINED"),
    "train-arm-b": ("ARM_A_TRAINED", "ARM_B_TRAINED"),
    "train-arm-c": ("ARM_B_TRAINED", "ARM_C_TRAINED"),
    "verify-training": ("ARM_C_TRAINED", "TRAINING_VERIFIED"),
    "evaluate-cloth": ("TRAINING_VERIFIED", "CLOTH_EVALUATED"),
    "evaluate-official": ("CLOTH_EVALUATED", "EVALUATED"),
    "verify-effect": ("EVALUATED", "COMPLETE"),
}

ARM_A_GATE_STOP_STATUS = "ARM_A_TRAINED_GATED"
RUNTIME_LIMIT_TRAIN_SECONDS = 900
RUNTIME_LIMIT_EVAL_SECONDS = 1800
MEMORY_RESERVE_GIB = 4
ENTRY_HEADROOM_GIB = 8
MAX_CPU_C = 90
MAX_GPU_C = 85

CRITERIA = [
    "primary: per-arm six-cloth mean RMSE change vs same-platform ZHY baseline E26 (0.1413798); four official per-joint <= +2%, world root / wrist <= +5% gates unchanged",
    "mechanism: arm B vs arm A six-cloth mean RMSE gap >= 3 percentage points in the less-degenerated direction -> projection mechanism established (H1 deepened); B ~= A -> Adam displacement is not the main cause, shift to H5",
    "arm A gate-hit optimizer step recorded; consistency with the E44/E45/HP12 ~19th-optimizer-step phenomenon is a recorded third reproduction, not a gate",
    "single-step attachment reading: if arm B first-step teacher RMSE change is still >0 at the E50 +1.38e-5 magnitude, record as the FP32 rounding floor; no mid-run redesign",
    "candidates are never deployed and never replace 035",
]
