"""Check the isolated EGL renderer without advancing simulation time."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mujoco
    import numpy as np
    from OpenGL import GL

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    with np.load(args.clip) as clip:
        data.qpos[:] = clip["init_qpos"]
        data.qvel[:] = clip["init_qvel"]
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        renderer._gl_context.make_current()
        strings = {k: GL.glGetString(v).decode() for k, v in
                   (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}
        if "NVIDIA" not in strings["vendor"]:
            raise RuntimeError("The isolated libraries did not produce an NVIDIA context")
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135.0, -20.0
        camera.lookat[:] = data.qpos[:3]
        started = time.monotonic()
        stds = []
        for _ in range(16):
            renderer.update_scene(data, camera)
            stds.append(float(renderer.render().std()))
        seconds = time.monotonic() - started
    if min(stds) <= 5 or data.time != 0:
        raise RuntimeError("Blank image or unexpected physics integration")
    result = dict(status="NVIDIA_EGL_RENDER_VERIFIED", **strings, frames=16,
        frame_size=[1280, 720], seconds=seconds, minimum_pixel_std=min(stds),
        physics_integrations=0, optimizer_updates=0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
