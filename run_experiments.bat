@echo off
setlocal EnableDelayedExpansion

REM ── Configuration ───────────────────────────────────────────────────────────
set PYTHON=python
set SCRIPT=raw_optimizer\synthetic_ct_dataset.py
set WIDTH=128
set HEIGHT=128
set N_ITER=100
set DEVICE=cuda

REM Meshes and shaders to run
set MESHES=sphere suzanne bunny
set SHADERS=ct_sh ct_env phong_sh phong_env

REM ── Phase 1: Generate dataset ────────────────────────────────────────────────
echo.
echo ============================================================
echo  Phase 1: Generating dataset (all meshes, all shaders)
echo ============================================================
for %%M in (%MESHES%) do (
    echo [Gen] mesh=%%M
    %PYTHON% %SCRIPT% --phase 1 --mesh %%M --shader all --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    if errorlevel 1 ( echo ERROR in Phase 1 for %%M & exit /b 1 )
)

REM ── Phase 2a: Full optimization (all params learnable) ───────────────────────
echo.
echo ============================================================
echo  Phase 2a: Full optimization (all params)
echo ============================================================
for %%M in (%MESHES%) do (
    for %%S in (%SHADERS%) do (
        echo [Full] mesh=%%M  shader=%%S
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR in Phase 2a for %%M %%S & exit /b 1 )
    )
)

REM ── Phase 2b: Ablation — single-param optimization ──────────────────────────
echo.
echo ============================================================
echo  Phase 2b: Ablation - single-param optimization
echo ============================================================

REM CT SH: albedo  sh  metallic  roughness
for %%M in (%MESHES%) do (
    for %%P in (albedo sh metallic roughness) do (
        echo [Ablation] mesh=%%M  shader=ct_sh  opt=%%P
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR & exit /b 1 )
    )
)

REM CT env: albedo  env  metallic  roughness
for %%M in (%MESHES%) do (
    for %%P in (albedo env metallic roughness) do (
        echo [Ablation] mesh=%%M  shader=ct_env  opt=%%P
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR & exit /b 1 )
    )
)

REM Phong SH: albedo  sh  shininess  ks
for %%M in (%MESHES%) do (
    for %%P in (albedo sh shininess ks) do (
        echo [Ablation] mesh=%%M  shader=phong_sh  opt=%%P
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR & exit /b 1 )
    )
)

REM Phong env: albedo  env  shininess  ks
for %%M in (%MESHES%) do (
    for %%P in (albedo env shininess ks) do (
        echo [Ablation] mesh=%%M  shader=phong_env  opt=%%P
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR & exit /b 1 )
    )
)

REM ── Phase 2c: Leave-one-out ablations ────────────────────────────────────────
echo.
echo ============================================================
echo  Phase 2c: Leave-one-out ablations (all-but-one param)
echo ============================================================

REM CT SH leave-one-out
for %%M in (%MESHES%) do (
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params sh,metallic,roughness    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params albedo,metallic,roughness --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params albedo,sh,roughness       --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params albedo,sh,metallic        --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
)

REM CT env leave-one-out
for %%M in (%MESHES%) do (
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params env,metallic,roughness    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params albedo,metallic,roughness --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params albedo,env,roughness      --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params albedo,env,metallic       --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
)

REM Phong SH leave-one-out
for %%M in (%MESHES%) do (
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params sh,shininess,ks        --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params albedo,shininess,ks    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params albedo,sh,ks           --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params albedo,sh,shininess    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
)

REM Phong env leave-one-out
for %%M in (%MESHES%) do (
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params env,shininess,ks      --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params albedo,shininess,ks   --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params albedo,env,ks         --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
    %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params albedo,env,shininess  --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
)

REM ── Phase 2d: lambda-white = 0.1 ────────────────────────────────────────────
echo.
echo ============================================================
echo  Phase 2d: lambda-white = 0.1
echo ============================================================
for %%M in (%MESHES%) do (
    for %%S in (%SHADERS%) do (
        echo [lw=0.1] mesh=%%M  shader=%%S
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-white 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR & exit /b 1 )
    )
)

REM ── Phase 2e: lambda-sparse = 0.1 ────────────────────────────────────────────
echo.
echo ============================================================
echo  Phase 2e: lambda-sparse = 0.1
echo ============================================================
for %%M in (%MESHES%) do (
    for %%S in (%SHADERS%) do (
        echo [ls=0.1] mesh=%%M  shader=%%S
        %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-sparse 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE%
        if errorlevel 1 ( echo ERROR & exit /b 1 )
    )
)

echo.
echo ============================================================
echo  All experiments complete.
echo ============================================================
