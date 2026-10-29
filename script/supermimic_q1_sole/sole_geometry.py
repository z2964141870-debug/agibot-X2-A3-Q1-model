"""Collision-sole geometry shared by independent Q1 reward candidates."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class FootSurface:
    body: str
    points: np.ndarray
    radii: np.ndarray
    geoms: tuple[str, ...] = ()

    def __post_init__(self):
        points = np.array(self.points, dtype=np.float64, copy=True)
        radii = np.array(self.radii, dtype=np.float64, copy=True)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or radii.shape != (len(points),)):
            raise ValueError("Expected nonempty local surface points and one radius per point")
        if not np.isfinite(points).all() or not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError("Surface geometry must be finite with nonnegative radii")
        points.setflags(write=False)
        radii.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "radii", radii)

    def identity(self):
        digest = hashlib.sha256()
        for value in (self.points, self.radii):
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.astype("<f8", copy=False).tobytes())
        return dict(body=self.body, points=len(self.points), geoms=list(self.geoms),
                    geometry_sha256=digest.hexdigest())


@dataclass(frozen=True)
class SoleGeometry:
    feet: tuple[FootSurface, FootSurface]

    def __post_init__(self):
        if tuple(foot.body for foot in self.feet) != FEET:
            raise ValueError("Expected exactly the left and right Q1 ankle-roll bodies in order")

    @classmethod
    def from_model(cls, model):
        import mujoco

        tools = Path(__file__).resolve().parents[1] / "project_tools"
        sys.path.insert(0, str(tools))
        from audit_q1_saved_sole_geometry import foot_geometry

        floor = model.geom("floor").id
        if (int(model.geom_type[floor]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
                or model.geom_bodyid[floor] != 0
                or not np.allclose(model.geom_pos[floor], 0.)
                or not np.allclose(model.geom_quat[floor], [1., 0., 0., 0.])):
            raise ValueError("This candidate requires the actual horizontal world floor at z0")
        feet = []
        for side in ("left", "right"):
            body, shapes = foot_geometry(model, side)
            points, radii, names = [], [], []
            for geom, kind, value in shapes:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                rotation = rotation.reshape(3, 3)
                if kind == "mesh":
                    points.append(value @ rotation.T + model.geom_pos[geom])
                    radii.append(np.zeros(len(value)))
                else:
                    # A sphere's lowest surface is center_z - radius in world Z.
                    points.append(model.geom_pos[geom][None].copy())
                    radii.append(np.array([value]))
                names.append(model.geom(geom).name)
            feet.append(FootSurface(model.body(body).name, np.concatenate(points),
                                    np.concatenate(radii), tuple(names)))
        return cls(tuple(feet))

    @classmethod
    def from_scene(cls, scene):
        import mujoco

        return cls.from_model(mujoco.MjModel.from_xml_path(str(scene)))

    def heights(self, positions, rotations):
        positions, rotations = np.asarray(positions), np.asarray(rotations)
        if (positions.shape[-2:] != (2, 3) or rotations.shape[-3:] != (2, 3, 3)
                or positions.shape[:-2] != rotations.shape[:-3]):
            raise ValueError("Expected matching foot positions (...,2,3) and rotations (...,2,3,3)")
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError("Foot poses must be finite")
        result = []
        for index, foot in enumerate(self.feet):
            z = np.einsum("...i,vi->...v", rotations[..., index, 2, :], foot.points)
            z = z + positions[..., index, 2, None] - foot.radii
            result.append(z.min(axis=-1))
        return np.stack(result, axis=-1)

    def identity(self):
        return dict(definition="Minimum active collision-mesh vertex or sphere surface world Z",
                    floor_z_m=0., feet=[foot.identity() for foot in self.feet])
