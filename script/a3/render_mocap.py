"""Render actual A3 meshes beside TEST_ONLY normalized body references."""

from io import BytesIO
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from script.a3.fullchain_support import DATA, mark
from script.a3.mocap_adapter import A3Retargeter, BODY_MAP, SOURCES


def main():
    links = [("pelvis", "spine3"), ("pelvis", "left_hip"), ("pelvis", "right_hip"),
             ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
             ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
             ("spine3", "left_shoulder"), ("spine3", "right_shoulder"),
             ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
             ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]
    names = list(BODY_MAP)
    outputs = []
    for name in SOURCES:
        directory = DATA / "mocap" / name
        with np.load(directory / "kinematic_trace.npz", allow_pickle=False) as trace:
            qpos, times = trace["qpos"], trace["time_s"]
            minimum = trace["foot_geometry_min_z_m"].min(axis=1)
        with np.load(directory / "body_reference.npz", allow_pickle=False) as body:
            positions = body["position_m_world"]
        solver = A3Retargeter()
        knee = solver.joint_names.index("left_knee_joint")
        picks = [0, int(np.argmax(qpos[:, solver.q_indices[knee]])), int(np.argmin(minimum)), len(times) - 1]
        montage = Image.new("RGB", (960, 4 * 460), "white")
        with mujoco.Renderer(solver.model, height=420, width=480) as renderer:
            for row, frame in enumerate(picks):
                points = positions[frame] - positions[frame, 0]
                fig = plt.figure(figsize=(4.8, 4.2), dpi=100)
                ax = fig.add_subplot(projection="3d")
                for a, b in links:
                    indices = [names.index(a), names.index(b)]
                    ax.plot(*points[indices].T, marker="o", markersize=3,
                            color="#267d82" if "left" in a + b else "#bd4652")
                ax.set(xlim=(-.9, .9), ylim=(-.9, .9), zlim=(-1.1, .8),
                       xlabel="Forward (m)", ylabel="Left (m)", zlabel="Up (m)")
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(15, 130)
                buffer = BytesIO()
                fig.savefig(buffer, format="png")
                plt.close(fig)
                montage.paste(Image.open(buffer).convert("RGB"), (0, row * 460 + 40))
                solver.data.qpos[:] = qpos[frame]
                mujoco.mj_forward(solver.model, solver.data)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = solver.data.xpos[solver.bodies[0]] + [0, 0, -.25]
                camera.distance, camera.azimuth, camera.elevation = 2.4, 130, -12
                renderer.update_scene(solver.data, camera=camera)
                montage.paste(Image.fromarray(renderer.render()), (480, row * 460 + 40))
                draw = ImageDraw.Draw(montage)
                draw.text((10, row * 460 + 10), f"{name} source t={times[frame]:.2f}s", fill="black")
                draw.text((490, row * 460 + 10), f"A3 IK only; foot min z={minimum[frame]:.3f}m", fill="black")
        output = directory / "body_a3_overview.png"
        montage.save(output)
        outputs.append(str(output))
    mark("retarget_visual", "passed", outputs=outputs, scope="rendered_not_pose_fidelity_acceptance")


if __name__ == "__main__":
    main()
