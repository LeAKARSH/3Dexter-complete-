export interface ModelRecord {
  id: string;
  name: string;
  prompt: string;
  route: "parametric" | "organic";
  createdAt: string;
  thumbnail?: string;
  version: number;
  code?: string;
  paramCount?: number;
  files?: string[];
  objFile?: string | null;
  plyFile?: string | null;
  objUrl?: string | null;
  plyUrl?: string | null;
  repairReports?: any[];
  modelType?: "shap-e" | "hunyuan3d";
  backend?: "local" | "gemini";
}

export interface ActivityRecord {
  id: string;
  type: "generated" | "exported";
  modelId: string;
  modelName: string;
  detail?: string;
  createdAt: string;
}

export type AppScreen = "home" | "editor";

export interface EditorEntry {
  model: ModelRecord | null;
  prompt?: string;
  route?: "parametric" | "organic";
  generationBackend?: "gemini";
}
