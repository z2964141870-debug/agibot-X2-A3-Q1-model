# Third-Party Sources

The `supermimic` package adapts the fixed local A3 implementation at repository
commit `538c8f3ee5f5dc7d6d33adaa8360b6ef21b7a480` to the supplied Q1 v3 model.
Its learning core comes from `CuboiLeo/SuperMimic-Reproduced`, commit
`d9be385b3eeb13a638fb33fb200d3bfd9be9839f`, MIT, copyright 2026 Yuxi Liu.
See `LICENSE.SuperMimic-Reproduced`. Robot dimensions and the physics/input
adapters are Q1 specific. No A3 or G1 learned weights are used.

`supermimic/kinematics/lab_math.py` retains the Isaac Lab BSD-3-Clause and
mjlab notices from `mujocolab/mjlab`, commit
`8bb2bc9e187797689ec82c78c35019afd2fd0376`.
See `LICENSE.IsaacLab-Math`.

The upstream `G1Kinematics` class name is preserved for provenance; this is
a generic compiled-MuJoCo hinge-tree FK and is separately verified on Q1.
Manufacturer meshes, XML, SDK, toolchains and motion inputs stay outside
Git. Their presence does not imply permission to redistribute them.

This is an independent method reproduction on Q1. It is not the authors'
official training release and does not reproduce their G1 dataset or timing.
