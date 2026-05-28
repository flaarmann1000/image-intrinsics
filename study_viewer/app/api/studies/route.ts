import { NextResponse } from "next/server";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import type { LightEntry, LightTypeEntry, RenderConfig, ResultEntry, SceneEntry } from "../../types";

const DATA_ROOT = path.resolve(/*turbopackIgnore: true*/ process.cwd(), "..", "synthetic_ct");
const DATASET_ROOT = path.join(DATA_ROOT, "dataset");
const RESULTS_ROOT = path.join(DATA_ROOT, "results");

async function exists(filePath: string) {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

async function listDirs(dirPath: string) {
  if (!(await exists(dirPath))) {
    return [];
  }

  return (await readdir(dirPath, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .sort((a, b) => a.name.localeCompare(b.name));
}

async function listPngs(dirPath: string) {
  if (!(await exists(dirPath))) {
    return [];
  }

  return (await readdir(dirPath, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".png"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));
}

async function readJson<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

function toAssetPath(absPath: string) {
  return path.relative(DATA_ROOT, absPath).split(path.sep).join("/");
}

async function optionalAsset(absPath: string) {
  return (await exists(absPath)) ? toAssetPath(absPath) : null;
}

function parseSceneId(sceneId: string) {
  const [mesh, maybeShader, ...materialParts] = sceneId.split("_");
  const shader = maybeShader === "phong" ? "phong" : "ct";

  return {
    mesh: mesh || sceneId,
    shader,
    materialId: shader === "phong"
      ? materialParts.join("_") || "default"
      : [maybeShader, ...materialParts].filter(Boolean).join("_") || "default"
  };
}

function parseAngle(lightId: string, config: RenderConfig | null) {
  if (typeof config?.light?.angle_deg === "number") {
    return config.light.angle_deg;
  }

  const match = lightId.match(/light_(\d+)deg/i);
  return match ? Number(match[1]) : null;
}

async function scanLightType(sceneDir: string, type: string, lightId: string): Promise<LightTypeEntry | null> {
  const typeDir = path.join(sceneDir, type, lightId);

  if (!(await exists(typeDir))) {
    return null;
  }

  const renderFile = path.join(typeDir, "render.png");
  const configFile = path.join(typeDir, "config.json");
  const componentsDir = path.join(typeDir, "components");
  const config = await readJson<RenderConfig>(configFile);
  const components = (await listPngs(componentsDir)).map((name) => `components/${name}`);

  return {
    type,
    renderPath: await optionalAsset(renderFile),
    configPath: await optionalAsset(configFile),
    components,
    config
  };
}

async function scanResults(sceneId: string): Promise<Record<string, ResultEntry>> {
  const sceneResultsDir = path.join(RESULTS_ROOT, sceneId);
  const methodDirs = await listDirs(sceneResultsDir);
  const results: Record<string, ResultEntry> = {};

  for (const methodDirEntry of methodDirs) {
    const method = methodDirEntry.name;
    const methodDir = path.join(sceneResultsDir, method);
    const metricsFile = path.join(methodDir, "metrics.json");
    const materialFile = path.join(methodDir, "material_est.json");
    const reconDir = path.join(methodDir, "reconstructions");
    const estimates: Record<string, string> = {};
    const reconstructions: Record<string, string> = {};
    const reconstructionErrors: Record<string, string> = {};

    for (const name of await listPngs(methodDir)) {
      estimates[name.replace(/\.png$/i, "")] = toAssetPath(path.join(methodDir, name));
    }

    for (const name of await listPngs(reconDir)) {
      const key = name.replace(/\.png$/i, "");
      const target = name.startsWith("recon_err_") ? reconstructionErrors : reconstructions;
      target[key] = toAssetPath(path.join(reconDir, name));
    }

    results[method] = {
      method,
      metricsPath: await optionalAsset(metricsFile),
      materialPath: await optionalAsset(materialFile),
      metrics: await readJson<Record<string, unknown>>(metricsFile),
      material: await readJson<Record<string, unknown>>(materialFile),
      estimates,
      reconstructions,
      reconstructionErrors
    };
  }

  return results;
}

async function scanScene(sceneDir: string, sceneId: string): Promise<SceneEntry> {
  const parsed = parseSceneId(sceneId);
  const renderTypeDirs = (await listDirs(sceneDir))
    .map((entry) => entry.name)
    .filter((name) => name !== "gt");

  const typeLightDirs = await Promise.all(
    renderTypeDirs.map(async (type) => ({
      type,
      lights: await listDirs(path.join(sceneDir, type))
    }))
  );

  const lightIds = Array.from(
    new Set(typeLightDirs.flatMap(({ lights }) => lights.map((entry) => entry.name)))
  ).sort((a, b) => a.localeCompare(b));

  const lights: LightEntry[] = [];
  let material = null;

  for (const lightId of lightIds) {
    const types: Record<string, LightTypeEntry> = {};
    let angleDeg: number | null = null;

    for (const type of renderTypeDirs) {
      const lightType = await scanLightType(sceneDir, type, lightId);
      if (!lightType) {
        continue;
      }

      types[type] = lightType;
      angleDeg ??= parseAngle(lightId, lightType.config);
      material ??= lightType.config?.material ?? null;
    }

    lights.push({ id: lightId, angleDeg, types });
  }

  const gtDir = path.join(sceneDir, "gt");
  const gtAssets: Record<string, string> = {};
  for (const name of await listPngs(gtDir)) {
    gtAssets[name.replace(/\.png$/i, "")] = toAssetPath(path.join(gtDir, name));
  }

  return {
    id: sceneId,
    mesh: parsed.mesh,
    shader: parsed.shader,
    materialId: parsed.materialId,
    material,
    groundTruth: {
      assets: gtAssets,
      albedoPath: await optionalAsset(path.join(gtDir, "albedo.png")),
      metallicPath: await optionalAsset(path.join(gtDir, "metallic.png")),
      normalsPath: await optionalAsset(path.join(gtDir, "normals.png")),
      roughnessPath: await optionalAsset(path.join(gtDir, "roughness.png"))
    },
    results: await scanResults(sceneId),
    lights
  };
}

export async function GET() {
  if (!(await exists(DATASET_ROOT))) {
    return NextResponse.json(
      {
        root: DATA_ROOT,
        datasetRoot: DATASET_ROOT,
        generatedAt: new Date().toISOString(),
        scenes: [],
        error: "synthetic_ct/dataset directory was not found next to study_viewer"
      },
      { status: 404 }
    );
  }

  const sceneDirs = await listDirs(DATASET_ROOT);
  const scenes = await Promise.all(
    sceneDirs.map((entry) => scanScene(path.join(DATASET_ROOT, entry.name), entry.name))
  );

  return NextResponse.json({
    root: DATA_ROOT,
    datasetRoot: DATASET_ROOT,
    resultsRoot: RESULTS_ROOT,
    generatedAt: new Date().toISOString(),
    scenes
  });
}
