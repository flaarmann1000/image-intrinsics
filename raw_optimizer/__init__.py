"""Optimizers and dataset drivers for CT/Phong intrinsic decomposition.

Intentionally free of imports. This module previously did

    from .optimizer import optimize
    from .scene_loader import load_scene, list_scenes

which pulled both (now-legacy) modules into memory on *every* import of any
submodule — including `raw_optimizer.dfront_ct`, the live path. Nothing used the
re-exported names, so they are gone; import submodules explicitly instead, e.g.

    from raw_optimizer.dfront_ct import load_scene, decompose_scene

Note that `dfront_ct.load_scene` is a *different* function from the old
`scene_loader.load_scene` this file used to export (now under legacy/misc/).
"""
