"use client";

import { ChevronLeft, ChevronRight, Eye, Grid3X3, ImageIcon, RefreshCw, SlidersHorizontal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { LightTypeEntry, ResultEntry, SceneEntry, StudyIndex } from "./types";

const ASSET_PREFIX = "/api/asset/";

function assetUrl(path: string | null | undefined) {
  return path ? `${ASSET_PREFIX}${path.split("/").map(encodeURIComponent).join("/")}` : "";
}

function formatNumber(value: unknown, digits = 4) {
  return typeof value === "number" ? value.toFixed(digits).replace(/\.?0+$/, "") : "n/a";
}

function vectorLabel(value: unknown) {
  return Array.isArray(value) ? value.map((item) => formatNumber(item, 2)).join(", ") : "n/a";
}

function prettyName(value: string) {
  return value.replaceAll("_", " ");
}

function metricLabel(value: string) {
  return value.replaceAll("_", " ");
}

function SceneThumb({ scene, selected }: { scene: SceneEntry; selected: boolean }) {
  const firstRender = Object.values(scene.lights[0]?.types ?? {}).find((entry) => entry.renderPath)?.renderPath;

  return (
    <>
      <div className="thumbFrame">
        {firstRender ? <img src={assetUrl(firstRender)} alt="" /> : <ImageIcon size={20} />}
      </div>
      <div className="sceneText">
        <span className="sceneName">{scene.id}</span>
        <span className="sceneMeta">
          {scene.mesh} / {scene.shader} / {scene.materialId}
        </span>
      </div>
      {selected ? <Eye size={16} className="selectedIcon" /> : null}
    </>
  );
}

function RenderCard({
  title,
  path,
  entry
}: {
  title: string;
  path?: string | null;
  entry?: LightTypeEntry;
}) {
  const image = path ?? entry?.renderPath ?? null;

  return (
    <figure className="renderCard">
      <div className="imageStage">
        {image ? (
          <img src={assetUrl(image)} alt={title} />
        ) : (
          <div className="missingImage">
            <ImageIcon size={26} />
            <span>Missing</span>
          </div>
        )}
      </div>
      <figcaption>{title}</figcaption>
    </figure>
  );
}

function MetricsTable({ result }: { result: ResultEntry | undefined }) {
  if (!result?.metrics) {
    return <p className="mutedText">No metrics for this result.</p>;
  }

  return (
    <dl className="metricsList">
      {Object.entries(result.metrics)
        .filter(([, value]) => typeof value === "number" || typeof value === "string")
        .map(([key, value]) => (
          <div key={key}>
            <dt>{metricLabel(key)}</dt>
            <dd>{typeof value === "number" ? formatNumber(value) : String(value)}</dd>
          </div>
        ))}
    </dl>
  );
}

export default function Home() {
  const [study, setStudy] = useState<StudyIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meshFilter, setMeshFilter] = useState("all");
  const [shaderFilter, setShaderFilter] = useState("all");
  const [materialFilter, setMaterialFilter] = useState("all");
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [lightIndex, setLightIndex] = useState(0);
  const [renderFamily, setRenderFamily] = useState<string | null>(null);
  const [componentType, setComponentType] = useState<string | null>(null);
  const [component, setComponent] = useState<string | null>(null);
  const [resultMethod, setResultMethod] = useState<string | null>(null);

  async function loadStudy() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/studies", { cache: "no-store" });
      const payload = (await response.json()) as StudyIndex & { error?: string };

      if (!response.ok) {
        throw new Error(payload.error ?? "Failed to load studies");
      }

      setStudy(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load studies");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadStudy();
  }, []);

  const scenes = study?.scenes ?? [];
  const meshOptions = useMemo(() => Array.from(new Set(scenes.map((scene) => scene.mesh))).sort(), [scenes]);
  const shaderOptions = useMemo(() => Array.from(new Set(scenes.map((scene) => scene.shader))).sort(), [scenes]);
  const materialOptions = useMemo(
    () =>
      Array.from(
        new Set(
          scenes
            .filter((scene) => shaderFilter === "all" || scene.shader === shaderFilter)
            .map((scene) => scene.materialId)
        )
      ).sort(),
    [scenes, shaderFilter]
  );

  const filteredScenes = useMemo(
    () =>
      scenes.filter((scene) => {
        return (
          (meshFilter === "all" || scene.mesh === meshFilter) &&
          (shaderFilter === "all" || scene.shader === shaderFilter) &&
          (materialFilter === "all" || scene.materialId === materialFilter)
        );
      }),
    [materialFilter, meshFilter, scenes, shaderFilter]
  );

  const selectedScene = useMemo(() => {
    return scenes.find((scene) => scene.id === sceneId) ?? filteredScenes[0] ?? scenes[0] ?? null;
  }, [filteredScenes, sceneId, scenes]);

  useEffect(() => {
    const selectedStillVisible = filteredScenes.some((scene) => scene.id === sceneId);
    const nextScene = selectedStillVisible ? null : filteredScenes[0] ?? scenes[0] ?? null;
    if (nextScene) {
      setSceneId(nextScene.id);
    }
  }, [filteredScenes, sceneId, scenes]);

  useEffect(() => {
    if (materialFilter !== "all" && !materialOptions.includes(materialFilter)) {
      setMaterialFilter("all");
    }
  }, [materialFilter, materialOptions]);

  useEffect(() => {
    setLightIndex(0);
    setComponent(null);
    setComponentType(null);
    setRenderFamily(null);
    setResultMethod(null);
  }, [selectedScene?.id]);

  const selectedLight = selectedScene?.lights[Math.min(lightIndex, Math.max((selectedScene?.lights.length ?? 1) - 1, 0))];
  const renderTypes = useMemo(() => Object.keys(selectedLight?.types ?? {}).sort(), [selectedLight]);
  const renderFamilies = useMemo(() => {
    const families = new Set<string>();
    for (const type of renderTypes) {
      if (type.endsWith("_sh")) {
        families.add(type.slice(0, -3));
      } else if (type.endsWith("_env")) {
        families.add(type.slice(0, -4));
      } else if (type === "sh" || type === "env") {
        families.add("");
      }
    }
    return Array.from(families).sort();
  }, [renderTypes]);
  const activeRenderFamily = renderFamily && renderFamilies.includes(renderFamily)
    ? renderFamily
    : renderFamilies[0] ?? "";
  const shType = activeRenderFamily ? `${activeRenderFamily}_sh` : "sh";
  const envType = activeRenderFamily ? `${activeRenderFamily}_env` : "env";
  const shEntry = selectedLight?.types[shType];
  const envEntry = selectedLight?.types[envType];
  const resultMethods = selectedScene ? Object.keys(selectedScene.results).sort() : [];
  const preferredResultMethod = resultMethods.find((method) => method === shType)
    ?? resultMethods.find((method) => method === activeRenderFamily)
    ?? resultMethods.find((method) => method.endsWith("_sh"))
    ?? resultMethods[0]
    ?? null;
  const activeResultMethod = resultMethod && resultMethods.includes(resultMethod) ? resultMethod : preferredResultMethod;
  const activeResult = activeResultMethod && selectedScene ? selectedScene.results[activeResultMethod] : undefined;
  const config = shEntry?.config ?? envEntry?.config;
  const activeComponentType = componentType && selectedLight?.types[componentType] ? componentType : shType;
  const componentEntry = selectedLight?.types[activeComponentType];
  const componentPath =
    component && selectedScene && selectedLight && componentEntry?.components.includes(component)
      ? `dataset/${selectedScene.id}/${activeComponentType}/${selectedLight.id}/${component}`
      : null;
  const groundTruthCards = Object.entries(selectedScene?.groundTruth.assets ?? {});
  const estimateCards = Object.entries(activeResult?.estimates ?? {})
    .filter(([key]) => key.endsWith("_est") || key === "albedo_est")
    .sort(([a], [b]) => a.localeCompare(b));
  const estimateErrorCards = Object.entries(activeResult?.estimates ?? {})
    .filter(([key]) => key.endsWith("_err"))
    .sort(([a], [b]) => a.localeCompare(b));

  useEffect(() => {
    if (activeResultMethod && !resultMethods.includes(activeResultMethod)) {
      setResultMethod(resultMethods[0] ?? null);
    }
  }, [activeResultMethod, resultMethods]);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brandBlock">
          <div className="brandIcon">
            <Grid3X3 size={18} />
          </div>
          <div>
            <h1>Synthetic CT Viewer</h1>
            <p>{scenes.length} scenes indexed</p>
          </div>
        </div>

        <section className="filterPanel">
          <div className="panelTitle">
            <SlidersHorizontal size={16} />
            <span>Filters</span>
          </div>
          <label>
            Mesh
            <select value={meshFilter} onChange={(event) => setMeshFilter(event.target.value)}>
              <option value="all">All meshes</option>
              {meshOptions.map((mesh) => (
                <option key={mesh} value={mesh}>
                  {mesh}
                </option>
              ))}
            </select>
          </label>
          <label>
            Shader
            <select value={shaderFilter} onChange={(event) => setShaderFilter(event.target.value)}>
              <option value="all">All shaders</option>
              {shaderOptions.map((shader) => (
                <option key={shader} value={shader}>
                  {shader.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          <label>
            Material
            <select value={materialFilter} onChange={(event) => setMaterialFilter(event.target.value)}>
              <option value="all">All materials</option>
              {materialOptions.map((material) => (
                <option key={material} value={material}>
                  {prettyName(material)}
                </option>
              ))}
            </select>
          </label>
        </section>

        <div className="sceneList" aria-label="Scenes">
          {filteredScenes.map((scene) => (
            <button
              key={scene.id}
              className={`sceneButton ${scene.id === selectedScene?.id ? "active" : ""}`}
              onClick={() => setSceneId(scene.id)}
            >
              <SceneThumb scene={scene} selected={scene.id === selectedScene?.id} />
            </button>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Study export</p>
            <h2>{selectedScene ? selectedScene.id : "No scene selected"}</h2>
          </div>
          <button className="iconButton" title="Refresh dataset index" onClick={() => void loadStudy()}>
            <RefreshCw size={17} />
          </button>
        </header>

        {loading ? <div className="emptyState">Loading synthetic_ct...</div> : null}
        {error ? <div className="emptyState errorState">{error}</div> : null}

        {selectedScene && selectedLight ? (
          <>
            <section className="controlBand twoColumnControls">
              <div className="lightStepper">
                <button
                  className="iconButton"
                  title="Previous light"
                  onClick={() => setLightIndex((value) => Math.max(value - 1, 0))}
                  disabled={lightIndex <= 0}
                >
                  <ChevronLeft size={18} />
                </button>
                <div className="angleReadout">
                  <span>{selectedLight.angleDeg ?? lightIndex} deg</span>
                  <small>
                    Light {lightIndex + 1} of {selectedScene.lights.length}
                  </small>
                </div>
                <button
                  className="iconButton"
                  title="Next light"
                  onClick={() => setLightIndex((value) => Math.min(value + 1, selectedScene.lights.length - 1))}
                  disabled={lightIndex >= selectedScene.lights.length - 1}
                >
                  <ChevronRight size={18} />
                </button>
              </div>

              <div className="angleStrip">
                {selectedScene.lights.map((light, index) => (
                  <button
                    key={light.id}
                    className={index === lightIndex ? "active" : ""}
                    onClick={() => setLightIndex(index)}
                  >
                    {light.angleDeg ?? light.id}
                  </button>
                ))}
              </div>
            </section>

            {renderFamilies.length > 1 ? (
              <section className="resultsHeader compactHeader">
                <div>
                  <h3>Render Shader</h3>
                  <p className="mutedText">Showing paired SH and ENV renders.</p>
                </div>
                <div className="segmented">
                  {renderFamilies.map((family) => (
                    <button
                      key={family || "default"}
                      className={activeRenderFamily === family ? "active" : ""}
                      onClick={() => {
                        setRenderFamily(family);
                        setComponentType(null);
                      }}
                    >
                      {(family || "render").toUpperCase()}
                    </button>
                  ))}
                </div>
              </section>
            ) : null}

            <section className="viewerGrid two">
              <RenderCard title={`${shType.toUpperCase()} render`} entry={shEntry} />
              <RenderCard title={`${envType.toUpperCase()} render`} entry={envEntry} />
            </section>

            <section className={`viewerGrid ${groundTruthCards.length > 2 ? "four" : "two"}`}>
              {groundTruthCards.map(([name, path]) => (
                <RenderCard key={name} title={`GT ${prettyName(name)}`} path={path} />
              ))}
              {groundTruthCards.length === 0 ? <RenderCard title="GT" /> : null}
            </section>

            <section className="resultsHeader">
              <div>
                <h3>Estimations</h3>
                <p className="mutedText">
                  {resultMethods.length > 0 ? `${resultMethods.length} result method(s)` : "No result folder for this scene."}
                </p>
              </div>
              {resultMethods.length > 0 ? (
                <div className="segmented">
                  {resultMethods.map((method) => (
                    <button
                      key={method}
                      className={activeResultMethod === method ? "active" : ""}
                      onClick={() => setResultMethod(method)}
                    >
                      {method.toUpperCase()}
                    </button>
                  ))}
                </div>
              ) : null}
            </section>

            <section className="viewerGrid four">
              {estimateCards.map(([name, path]) => (
                <RenderCard key={name} title={prettyName(name)} path={path} />
              ))}
              {estimateErrorCards.map(([name, path]) => (
                <RenderCard key={name} title={prettyName(name)} path={path} />
              ))}
              {estimateCards.length + estimateErrorCards.length === 0 ? <RenderCard title="Estimates" /> : null}
            </section>

            <section className="viewerGrid two">
              <RenderCard
                title="Reconstruction"
                path={activeResult?.reconstructions[`recon_${String(selectedLight.angleDeg ?? 0).padStart(2, "0")}deg`]}
              />
              <RenderCard
                title="Selected component"
                path={componentPath}
              />
            </section>

            <section className="detailsGrid">
              <div className="detailPanel">
                <h3>Material</h3>
                <dl>
                  <div>
                    <dt>Albedo</dt>
                    <dd>{vectorLabel(selectedScene.material?.albedo)}</dd>
                  </div>
                  <div>
                    <dt>{selectedScene.shader === "phong" ? "Ks" : "Metallic"}</dt>
                    <dd>{formatNumber(selectedScene.material?.metallic ?? selectedScene.material?.ks)}</dd>
                  </div>
                  <div>
                    <dt>{selectedScene.shader === "phong" ? "Shininess" : "Roughness"}</dt>
                    <dd>{formatNumber(selectedScene.material?.roughness ?? selectedScene.material?.shininess)}</dd>
                  </div>
                </dl>
              </div>

              <div className="detailPanel">
                <h3>Light</h3>
                <dl>
                  <div>
                    <dt>Direction</dt>
                    <dd>{vectorLabel(config?.light?.direction)}</dd>
                  </div>
                  <div>
                    <dt>Color</dt>
                    <dd>{vectorLabel(config?.light?.color)}</dd>
                  </div>
                  <div>
                    <dt>Intensity</dt>
                    <dd>{formatNumber(config?.light?.intensity)}</dd>
                  </div>
                </dl>
              </div>

              <div className="detailPanel">
                <h3>Metrics</h3>
                <MetricsTable result={activeResult} />
              </div>
            </section>

            <section className="detailPanel componentsPanel">
              <div className="componentHeader">
                <h3>Components</h3>
                <div className="segmented">
                  {renderTypes.map((type) => (
                    <button
                      key={type}
                      className={activeComponentType === type ? "active" : ""}
                      onClick={() => {
                        setComponentType(type);
                      }}
                      disabled={!selectedLight.types[type]}
                    >
                      {type.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
              <div className="componentList">
                {(componentEntry?.components ?? []).map((name) => (
                  <button
                    key={name}
                    className={component === name ? "active" : ""}
                    onClick={() => setComponent((current) => (current === name ? null : name))}
                  >
                    {name.replace("components/", "").replace(".png", "")}
                  </button>
                ))}
                {(componentEntry?.components.length ?? 0) === 0 ? <p className="mutedText">No components found.</p> : null}
              </div>
            </section>
          </>
        ) : null}
      </section>
    </main>
  );
}
