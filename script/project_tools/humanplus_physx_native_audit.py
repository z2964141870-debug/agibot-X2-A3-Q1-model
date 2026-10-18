"""Archive HP09 native exit evidence without starting any workload."""

from script.a3 import humanplus_physx_native_validation as runtime
from script.project_tools.humanplus_physx_exit_audit import main


if __name__ == '__main__':
    main(runtime=runtime)
