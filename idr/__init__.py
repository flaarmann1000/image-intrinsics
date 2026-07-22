"""idr — intrinsic decomposition and relighting.

Layout:

    render/     GPU renderer: shading models, BRDF, SH, rasterisation
    data/       scene IO, proxy geometry, dataset building
    optim/      the optimizers, their parameter transforms, losses and LM solvers
    eval/       metrics, relighting evaluation, artifacts and plots
    track/      experiment logging
    pipelines/  end-to-end drivers (decompose a scene, build+run a synthetic study)

Submodules are imported explicitly — this file deliberately stays free of imports so
that pulling in one subpackage does not drag in the rest.
"""
