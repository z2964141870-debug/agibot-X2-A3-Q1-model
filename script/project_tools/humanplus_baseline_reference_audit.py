"""Run the frozen standard-library auditor against independently registered HP11."""

import argparse

from script.a3 import humanplus_baseline_reference as trial
from script.project_tools import humanplus_baseline_audit as auditor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=trial.STAGES, required=True)
    args = parser.parse_args()
    auditor.trial = trial
    auditor.main(args.stage)
