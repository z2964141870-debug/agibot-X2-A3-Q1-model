#!/usr/bin/env python3
"""Run the preserved partial-parent loop with a matched Q1 action domain."""

from functools import partial

from run_contract import HERE, parent, correction_declaration
import train_partial
from bounded_policy import BoundedPolicy, GuardedPolicyTrainer, GuardedWorldModelTrainer


def registered_guards(optimizers):
    expected = "reject_nonfinite_gradients.<locals>.before_step"
    result = {}
    for label, optimizer in zip(parent.OPTIMIZER_KEYS, optimizers):
        hooks = list(optimizer._optimizer_step_pre_hooks.values())
        if len(hooks) != 1 or hooks[0].__module__ != "bounded_policy" or hooks[0].__qualname__ != expected:
            raise ValueError("Required nonfinite-gradient guard is not registered: " + label)
        result[label] = {"registered_pre_step_hooks": 1,
                         "callback": hooks[0].__module__ + "." + hooks[0].__qualname__}
    return result


class BoundedInitialization:
    def __getattr__(self, name):
        return getattr(parent, name)

    def load_parent(self, evidence, target, modules, optimizers, sim):
        if not isinstance(modules[1], BoundedPolicy):
            raise ValueError("The actual training policy must apply clip20")
        before = registered_guards(optimizers)
        result = parent.load_parent(evidence, target, modules, optimizers, sim)
        if registered_guards(optimizers) != before:
            raise ValueError("Loading parent Adam states removed a required step guard")
        correction = {**correction_declaration(), "runtime_optimizer_guards": before,
                      "runtime_policy_class": type(modules[1]).__module__ + "." + type(modules[1]).__name__}
        target["action_domain_correction"] = correction
        target["training_entry"] = parent.identity(__file__)
        sources = {row["path"]: row for row in target["source_files"]}
        for name in ("train_bounded.py", "run_contract.py", "bounded_policy.py"):
            record = parent.identity(HERE / name)
            sources[record["path"]] = record
        target["source_files"] = [sources[key] for key in sorted(sources)]
        target["paper_deviations"].append(
            "Real and imagined clean actions share clip20; nonfinite raw actions or pre-Adam gradients stop the trial.")
        result["action_domain_correction"] = correction
        return result


def main():
    trainer = train_partial.trainer
    if trainer.Q1Sim.__name__ != "Q1Sim" or trainer.Policy.__name__ != "Policy":
        raise ValueError("Unexpected preserved trainer bindings")
    trainer.initialization = BoundedInitialization()
    trainer.Q1Sim = partial(train_partial.SlottedQ1Sim, slots=4)
    trainer.Policy = BoundedPolicy
    trainer.PolicyTrainer = GuardedPolicyTrainer
    trainer.WorldModelTrainer = GuardedWorldModelTrainer
    trainer.main()


if __name__ == "__main__":
    main()
