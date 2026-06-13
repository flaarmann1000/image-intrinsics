@echo off
setlocal EnableDelayedExpansion

REM ── Configuration ───────────────────────────────────────────────────────────
set PYTHON=python
set SCRIPT=raw_optimizer\synthetic_ct_dataset.py
set WIDTH=128
set HEIGHT=128
set N_ITER=100
set DEVICE=cuda
set SKIP=--skip-existing

REM Meshes and material configs
@REM set MESHES=sphere suzanne bunny
set MESHES=sphere
@REM set MAT_CONFIGS=  (empty = all)
set MAT_CONFIGS=albedo_0

REM T : parameter domain transforms  (none | all | custom)
set T=none

REM L : lighting model               (sh | env | all)
set L=all

REM Lighting configuration
set N_LIGHTS=6
set LIGHT_MODE=directional
set FULL_CIRCLE=0
set INIT_FROM_GT=0
set LOG_GRADIENTS=0

REM ── Sub-phases to run (comment out to skip) ─────────────────────────────────
@REM set RUN_1=1        & REM  dataset generation
set RUN_2A=1           & REM  full optimization (all params)
set RUN_2B=1           & REM  single-param ablations
@REM set RUN_2C=1       & REM  leave-one-out ablations
set RUN_2_SH_LW=1      & REM  SH  + lw=0.1
set RUN_2_ENV_LW=1     & REM  ENV + lw=0.1
set RUN_2_SH_LS=1      & REM  SH  + ls=0.1
set RUN_2_ENV_LS=1     & REM  ENV + ls=0.1
set RUN_2_SH_LT=1      & REM  SH  + lt=0.1
set RUN_2_ENV_LT=1     & REM  ENV + lt=0.1

REM ── Derived variables ────────────────────────────────────────────────────────
set TRANSFORMS=%T%
if defined MAT_CONFIGS (set MC_FLAG=--mat-configs %MAT_CONFIGS%) else (set MC_FLAG=)
if defined TRANSFORMS  (set TR_FLAG=--transforms %TRANSFORMS%)   else (set TR_FLAG=)

set LIGHT_ARGS=--n-lights %N_LIGHTS% --light-mode %LIGHT_MODE%
if %FULL_CIRCLE%==1   set LIGHT_ARGS=%LIGHT_ARGS% --full-circle
if %INIT_FROM_GT%==1  set LIGHT_ARGS=%LIGHT_ARGS% --init-from-gt
if %LOG_GRADIENTS%==1 set LIGHT_ARGS=%LIGHT_ARGS% --log-gradients

set SH_SHADERS=ct_sh phong_sh
set ENV_SHADERS=ct_env phong_env
if /i "%L%"=="sh"  set ENV_SHADERS=
if /i "%L%"=="env" set SH_SHADERS=
set SHADERS=%SH_SHADERS% %ENV_SHADERS%

set /a N_SH=2 & set /a N_ENV=2
if /i "%L%"=="sh"  set /a N_ENV=0
if /i "%L%"=="env" set /a N_SH=0
set /a N_SHADERS=N_SH+N_ENV

REM ── Compute total run count (only enabled phases) ───────────────────────────
set /a N_MESHES=0
for %%M in (%MESHES%) do set /a N_MESHES+=1

set /a TOTAL_RUNS=0
if defined RUN_1 (
    set /a TOTAL_RUNS+=1*N_MESHES
)
if defined RUN_2A (
    set /a TOTAL_RUNS+=N_SHADERS*N_MESHES
)
if defined RUN_2B (
    set /a _t=N_SH*4+N_ENV*4
    set /a TOTAL_RUNS+=_t*N_MESHES
)
if defined RUN_2C (
    set /a _t=N_SH*4+N_ENV*4
    set /a TOTAL_RUNS+=_t*N_MESHES
)
if defined RUN_2_SH_LW  set /a TOTAL_RUNS+=N_SH*N_MESHES
if defined RUN_2_ENV_LW set /a TOTAL_RUNS+=N_ENV*N_MESHES
if defined RUN_2_SH_LS  set /a TOTAL_RUNS+=N_SH*N_MESHES
if defined RUN_2_ENV_LS set /a TOTAL_RUNS+=N_ENV*N_MESHES
if defined RUN_2_SH_LT  set /a TOTAL_RUNS+=N_SH*N_MESHES
if defined RUN_2_ENV_LT set /a TOTAL_RUNS+=N_ENV*N_MESHES

set /a RUN_NUM=0

REM ── Config note (only active axes) ──────────────────────────────────────────
set _NOTE=
if defined RUN_1        set _NOTE=!_NOTE! 1
if defined RUN_2A       set _NOTE=!_NOTE! 2A
if defined RUN_2B       set _NOTE=!_NOTE! 2B
if defined RUN_2C       set _NOTE=!_NOTE! 2C
if defined RUN_2_SH_LW  if %N_SH% gtr 0 set _NOTE=!_NOTE! SH+LW
if defined RUN_2_ENV_LW if %N_ENV% gtr 0 set _NOTE=!_NOTE! ENV+LW
if defined RUN_2_SH_LS  if %N_SH% gtr 0 set _NOTE=!_NOTE! SH+LS
if defined RUN_2_ENV_LS if %N_ENV% gtr 0 set _NOTE=!_NOTE! ENV+LS
if defined RUN_2_SH_LT  if %N_SH% gtr 0 set _NOTE=!_NOTE! SH+LT
if defined RUN_2_ENV_LT if %N_ENV% gtr 0 set _NOTE=!_NOTE! ENV+LT

echo.
echo  Total runs : %TOTAL_RUNS%  (across %N_MESHES% mesh(es))
echo  T=%T%   L=%L%   light-mode=%LIGHT_MODE%   n-lights=%N_LIGHTS%   full-circle=%FULL_CIRCLE%   init-from-gt=%INIT_FROM_GT%   log-gradients=%LOG_GRADIENTS%
echo  phases:%_NOTE%
echo.

REM ── Phase 1: Generate dataset ────────────────────────────────────────────────
if defined RUN_1 (
    echo ============================================================
    echo  Phase 1: Generating dataset
    echo ============================================================
    for %%M in (%MESHES%) do (
        set /a RUN_NUM+=1
        echo [!RUN_NUM!/%TOTAL_RUNS%] Gen  mesh=%%M
        %PYTHON% %SCRIPT% --phase 1 --mesh %%M --shader all --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
        if errorlevel 1 ( echo ERROR in Phase 1 for %%M & exit /b 1 )
    )
)

REM ── Phase 2a: Full optimization ──────────────────────────────────────────────
if defined RUN_2A (
    echo.
    echo ============================================================
    echo  Phase 2a: Full optimization (all params)
    echo ============================================================
    for %%M in (%MESHES%) do (
        for %%S in (%SHADERS%) do (
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] Full  mesh=%%M  shader=%%S
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            if errorlevel 1 ( echo ERROR in Phase 2a for %%M %%S & exit /b 1 )
        )
    )
)

REM ── Phase 2b: Single-param ablations ─────────────────────────────────────────
if defined RUN_2B (
    echo.
    echo ============================================================
    echo  Phase 2b: Single-param ablations
    echo ============================================================

    if %N_SH% gtr 0 (
        for %%M in (%MESHES%) do (
            for %%P in (albedo sh metallic roughness) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] ct_sh  opt=%%P  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
        for %%M in (%MESHES%) do (
            for %%P in (albedo sh shininess ks) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] phong_sh  opt=%%P  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )

    if %N_ENV% gtr 0 (
        for %%M in (%MESHES%) do (
            for %%P in (albedo env metallic roughness) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] ct_env  opt=%%P  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
        for %%M in (%MESHES%) do (
            for %%P in (albedo env shininess ks) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] phong_env  opt=%%P  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params %%P --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

REM ── Phase 2c: Leave-one-out ablations ────────────────────────────────────────
if defined RUN_2C (
    echo.
    echo ============================================================
    echo  Phase 2c: Leave-one-out ablations
    echo ============================================================

    if %N_SH% gtr 0 (
        for %%M in (%MESHES%) do (
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_sh  opt=sh,metallic,roughness  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params sh,metallic,roughness    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_sh  opt=albedo,metallic,roughness  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params albedo,metallic,roughness --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_sh  opt=albedo,sh,roughness  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params albedo,sh,roughness       --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_sh  opt=albedo,sh,metallic  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_sh --opt-params albedo,sh,metallic        --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
        )
        for %%M in (%MESHES%) do (
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_sh  opt=sh,shininess,ks  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params sh,shininess,ks        --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_sh  opt=albedo,shininess,ks  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params albedo,shininess,ks    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_sh  opt=albedo,sh,ks  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params albedo,sh,ks           --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_sh  opt=albedo,sh,shininess  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_sh --opt-params albedo,sh,shininess    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
        )
    )

    if %N_ENV% gtr 0 (
        for %%M in (%MESHES%) do (
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_env  opt=env,metallic,roughness  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params env,metallic,roughness    --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_env  opt=albedo,metallic,roughness  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params albedo,metallic,roughness --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_env  opt=albedo,env,roughness  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params albedo,env,roughness      --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] ct_env  opt=albedo,env,metallic  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader ct_env --opt-params albedo,env,metallic       --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
        )
        for %%M in (%MESHES%) do (
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_env  opt=env,shininess,ks  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params env,shininess,ks      --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_env  opt=albedo,shininess,ks  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params albedo,shininess,ks   --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_env  opt=albedo,env,ks  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params albedo,env,ks         --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
            set /a RUN_NUM+=1
            echo [!RUN_NUM!/%TOTAL_RUNS%] phong_env  opt=albedo,env,shininess  mesh=%%M
            %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader phong_env --opt-params albedo,env,shininess  --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
        )
    )
)

REM ── Phase 2 SH + lambda-white ────────────────────────────────────────────────
if defined RUN_2_SH_LW (
    if %N_SH% gtr 0 (
        echo.
        echo ============================================================
        echo  Phase 2: SH + lw=0.1
        echo ============================================================
        for %%M in (%MESHES%) do (
            for %%S in (%SH_SHADERS%) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] %%S  lw=0.1  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-white 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

REM ── Phase 2 ENV + lambda-white ───────────────────────────────────────────────
if defined RUN_2_ENV_LW (
    if %N_ENV% gtr 0 (
        echo.
        echo ============================================================
        echo  Phase 2: ENV + lw=0.1
        echo ============================================================
        for %%M in (%MESHES%) do (
            for %%S in (%ENV_SHADERS%) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] %%S  lw=0.1  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-white 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

REM ── Phase 2 SH + lambda-sparse ───────────────────────────────────────────────
if defined RUN_2_SH_LS (
    if %N_SH% gtr 0 (
        echo.
        echo ============================================================
        echo  Phase 2: SH + ls=0.1
        echo ============================================================
        for %%M in (%MESHES%) do (
            for %%S in (%SH_SHADERS%) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] %%S  ls=0.1  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-sparse 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

REM ── Phase 2 ENV + lambda-sparse ──────────────────────────────────────────────
if defined RUN_2_ENV_LS (
    if %N_ENV% gtr 0 (
        echo.
        echo ============================================================
        echo  Phase 2: ENV + ls=0.1
        echo ============================================================
        for %%M in (%MESHES%) do (
            for %%S in (%ENV_SHADERS%) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] %%S  ls=0.1  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-sparse 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

REM ── Phase 2 SH + lambda-tv ────────────────────────────────────────────────────
if defined RUN_2_SH_LT (
    if %N_SH% gtr 0 (
        echo.
        echo ============================================================
        echo  Phase 2: SH + lt=0.1
        echo ============================================================
        for %%M in (%MESHES%) do (
            for %%S in (%SH_SHADERS%) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] %%S  lt=0.1  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-tv 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

REM ── Phase 2 ENV + lambda-tv ───────────────────────────────────────────────────
if defined RUN_2_ENV_LT (
    if %N_ENV% gtr 0 (
        echo.
        echo ============================================================
        echo  Phase 2: ENV + lt=0.1
        echo ============================================================
        for %%M in (%MESHES%) do (
            for %%S in (%ENV_SHADERS%) do (
                set /a RUN_NUM+=1
                echo [!RUN_NUM!/%TOTAL_RUNS%] %%S  lt=0.1  mesh=%%M
                %PYTHON% %SCRIPT% --phase 2 --mesh %%M --shader %%S --lambda-tv 0.1 --n-iter %N_ITER% --width %WIDTH% --height %HEIGHT% --device %DEVICE% %SKIP% %MC_FLAG% %TR_FLAG% %LIGHT_ARGS%
                if errorlevel 1 ( echo ERROR & exit /b 1 )
            )
        )
    )
)

echo.
echo ============================================================
echo  All %TOTAL_RUNS% runs complete.
echo ============================================================
