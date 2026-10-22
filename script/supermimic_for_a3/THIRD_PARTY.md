# Third-Party Sources

The `supermimic` package is an A3 adaptation of selected modules from
`https://github.com/CuboiLeo/SuperMimic-Reproduced`, commit
`d9be385b3eeb13a638fb33fb200d3bfd9be9839f`, MIT, copyright 2026 Yuxi Liu.
See `LICENSE.SuperMimic-Reproduced`. Modified modules retain the upstream
algorithm and add full A3 joint state, active-joint observation selection,
explicit episode boundaries, and a MuJoCo CPU transition backend.

`supermimic/kinematics/lab_math.py` is copied from
`mujocolab/mjlab`, commit `8bb2bc9e187797689ec82c78c35019afd2fd0376`,
`src/mjlab/utils/lab_api/math.py`. Its retained header identifies Isaac Lab
BSD-3-Clause code and subsequent mjlab modifications.
See `LICENSE.IsaacLab-Math`.

The robot model and motion inputs remain in the existing server asset tree.
They are neither embedded nor redistributed by this directory. Runtime manifests
identify their exact hashes and paths. The paper's proprietary browser demo,
weights, and motion assets are not used.

This is an independent A3 adaptation. Upstream G1 metrics are not our results.
