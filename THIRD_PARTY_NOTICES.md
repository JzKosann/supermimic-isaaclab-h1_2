# Third-Party Notices

This repository contains or is derived from the following third-party work.

## Code

### SuperMimic-Reproduced — MIT License

<https://github.com/CuboiLeo/SuperMimic-Reproduced>

The world model, imagination trainer, analytic heads, replay buffer, return-based reset sampling and
benchmark in `source/supermimic_h1_2/supermimic_h1_2/mdp/` were ported module by module from this
mjlab / Unitree G1 reproduction to Isaac Lab / Unitree H1-2.

```
MIT License

Copyright (c) 2026 Yuxi Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### supermimic_vibed — MIT License

<https://github.com/zita-ch/supermimic_vibed>

The `v1` recipe (reference root-orientation observation, trajectory-style goal, world-model warm-up,
clip-end-aware imagination starts, analytic soft termination, tracking error in the critic input)
follows this Isaac Lab 3 / Unitree G1 reproduction.

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### unitree_rl_lab — Apache License 2.0

<https://github.com/unitreerobotics/unitree_rl_lab>

The H1-2 actuator groups, default joint pose and solver settings in
`source/supermimic_h1_2/supermimic_h1_2/robots.py` are taken from
`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
Copyright Unitree Robotics. Licensed under the Apache License, Version 2.0
(<http://www.apache.org/licenses/LICENSE-2.0>).

### Isaac Lab — BSD 3-Clause License

<https://github.com/isaac-sim/IsaacLab>

The extension layout (`source/supermimic_h1_2/setup.py`, `config/extension.toml`) follows the Isaac Lab
extension template. Copyright (c) 2022-2025, The Isaac Lab Project Developers.

## Robot model

### unitree_ros — BSD 3-Clause License

<https://github.com/unitreerobotics/unitree_ros>

`assets/h1_2_description/h1_2_handless.urdf` is copied unchanged from
`robots/h1_2_description/h1_2_handless.urdf`. Its license is in `assets/h1_2_description/LICENSE`.

### unitree_model (not redistributed)

<https://huggingface.co/datasets/unitreerobotics/unitree_model>

The H1-2 USD is downloaded by `scripts/download_assets.py` and is not part of this repository.

## Motion data (not redistributed)

- **LAFAN1** — Ubisoft La Forge Animation Dataset, <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>.
  Please follow its license (CC BY-NC-ND 4.0).
- **LAFAN1 Retargeting Dataset** (H1-2 retargeting) — <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>.

The CSV files are downloaded by `scripts/download_assets.py`; neither the CSVs nor the converted npz
files are part of this repository.
