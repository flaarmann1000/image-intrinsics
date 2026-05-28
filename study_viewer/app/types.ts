export type MaterialConfig = {
  albedo?: number[];
  metallic?: number;
  roughness?: number;
  [key: string]: unknown;
};

export type LightConfig = {
  angle_deg?: number;
  direction?: number[];
  color?: number[];
  intensity?: number;
  sh_coeffs?: number[][] | number[][][];
};

export type RenderConfig = {
  mesh_name?: string;
  material?: MaterialConfig;
  light?: LightConfig;
  render_resolution?: number[];
  light_type?: string;
};

export type LightTypeEntry = {
  type: "sh" | "env" | string;
  renderPath: string | null;
  configPath: string | null;
  components: string[];
  config: RenderConfig | null;
};

export type GroundTruthEntry = {
  assets: Record<string, string>;
  albedoPath: string | null;
  metallicPath: string | null;
  normalsPath: string | null;
  roughnessPath: string | null;
};

export type ResultEntry = {
  method: string;
  metricsPath: string | null;
  materialPath: string | null;
  metrics: Record<string, unknown> | null;
  material: Record<string, unknown> | null;
  estimates: Record<string, string>;
  reconstructions: Record<string, string>;
  reconstructionErrors: Record<string, string>;
};

export type LightEntry = {
  id: string;
  angleDeg: number | null;
  types: Record<string, LightTypeEntry>;
};

export type SceneEntry = {
  id: string;
  mesh: string;
  shader: string;
  materialId: string;
  material: MaterialConfig | null;
  groundTruth: GroundTruthEntry;
  results: Record<string, ResultEntry>;
  lights: LightEntry[];
};

export type StudyIndex = {
  root: string;
  generatedAt: string;
  scenes: SceneEntry[];
};
