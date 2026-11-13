"""Batch independent joints at each SMPL tree depth; retain the original math."""
import numpy as np

from script.x2_morph_human.core import PARENTS, SmplAdapter, array, mul, rotate, unit


class BatchedSmplAdapter(SmplAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        depths = [0] * 24
        for i in range(1, 24):
            depths[i] = depths[PARENTS[i]] + 1
        self.levels = [np.flatnonzero(np.asarray(depths) == d)
                       for d in range(1, max(depths) + 1)]
        self.parents = np.asarray(PARENTS)

    def _fk_from_local_quats(self, local_quat_24x4, root_translation_m):
        world = np.empty((24, 4))
        c = unit(np.asarray(self.calibration.world_from_smpl_wxyz))
        world[0] = mul(c, local_quat_24x4[0])
        for indices in self.levels:
            world[indices] = mul(world[self.parents[indices]], local_quat_24x4[indices])
        root = rotate(c, array(root_translation_m, (3,), "root_translation_m"))
        root = root * self.calibration.translation_scale + self.calibration.world_offset_m
        displacement = rotate(world[self.parents[1:]], self.bones[1:])
        pos = np.empty((24, 3))
        pos[0] = root
        for indices in self.levels:
            pos[indices] = pos[self.parents[indices]] + displacement[indices - 1]
        return pos, unit(world)
