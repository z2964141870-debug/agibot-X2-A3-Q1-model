"""Accept only nonlinear IK integration steps whose actual feet clear the floor."""

import mink
import numpy as np

from script.a3.fullchain_support import write_json
from script.a3.smpl_collision_recovery import RecoveringA3Retargeter


class FloorCheckedConfiguration(mink.Configuration):
    def __init__(self, model, q, foot_geometries):
        super().__init__(model, q=q)
        self.foot_geometries = foot_geometries
        self.integration_records = []
        if self.minimum_foot_z() < 0:
            raise ValueError('Initialization must already clear the actual floor')

    def minimum_foot_z(self):
        return min(float((vertices @ self.data.geom_xmat[geom].reshape(3, 3).T
                          + self.data.geom_xpos[geom])[:, 2].min())
                   for side in self.foot_geometries for geom, vertices in side)

    def integrate_inplace(self, velocity, dt):
        before = self.data.qpos.copy()
        start_z = self.minimum_foot_z()
        if start_z < 0:
            raise ValueError('Accepted state already violates floor geometry')
        full_candidate_z = None
        accepted_factor = 0.
        for trial in range(13):
            factor = 2. ** -trial
            self.update(before)
            super().integrate_inplace(velocity * factor, dt)
            candidate_z = self.minimum_foot_z()
            if full_candidate_z is None:
                full_candidate_z = candidate_z
            if candidate_z >= 0:
                accepted_factor = factor
                break
        if accepted_factor == 0:
            self.update(before)
        self.integration_records.append([start_z, full_candidate_z,
            self.minimum_foot_z(), accepted_factor, trial + 1,
            float(np.max(np.abs(velocity * dt)))])


class CheckedA3Retargeter(RecoveringA3Retargeter):
    def __init__(self):
        super().__init__()
        self.configuration = FloorCheckedConfiguration(self.model,
            self.configuration.data.qpos.copy(), self.foot_geometries)

    def solve(self, source, scale, directory):
        self.configuration.integration_records.clear()
        summary = super().solve(source, scale, directory)
        records = np.asarray(self.configuration.integration_records)
        np.savez_compressed(directory / 'integration_checks.npz', records=records,
            columns=np.asarray(['start_min_z_m', 'full_candidate_min_z_m',
                'accepted_min_z_m', 'accepted_factor', 'attempts', 'full_max_delta_q']),
            usage=np.asarray('NONLINEAR_IK_STEP_QUALIFICATION'))
        summary.update(nonlinear_floor_checked=True, accepted_floor_min_m=float(records[:, 2].min()),
            integration_steps=len(records), reduced_steps=int(np.count_nonzero(records[:, 3] < 1)),
            held_steps=int(np.count_nonzero(records[:, 3] == 0)),
            full_candidate_floor_min_m=float(records[:, 1].min()),
            extra_simulation_time_added=False)
        write_json(directory / 'retarget_summary.json', summary)
        return summary
