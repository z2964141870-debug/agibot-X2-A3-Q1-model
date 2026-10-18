"""Archive HP08 using the shared terminal PhysX evidence format."""

from script.a3 import humanplus_physx_callback_validation as runtime
from script.project_tools.humanplus_physx_exit_audit import main


if __name__ == '__main__':
    main(runtime=runtime)
