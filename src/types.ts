/** Shared plain-data types. Every compute function takes and returns these — no classes, no I/O. */

/** A normalized `rope_scaling` block as found in a config.json. */
export interface RopeScalingBlock {
  /** Normalized scaling type: `rope_type` wins over legacy `type` when both are present. */
  type: string;
  /** True when the block used the legacy `type` key (pre-4.43 transformers style). */
  usedLegacyKey: boolean;
  /** True when both `type` and `rope_type` were present with different values. */
  conflictingTypeKeys: boolean;
  factor?: number;
  originalMaxPositionEmbeddings?: number;
  betaFast?: number;
  betaSlow?: number;
  attentionFactor?: number;
  lowFreqFactor?: number;
  highFreqFactor?: number;
  /** Keys present with a recognized name but a non-numeric / non-finite value. */
  invalidKeys: string[];
  /** Recognized runtime-fork keys (e.g. DeepSeek's mscale) present in the block. */
  forkKeys: string[];
  /** Keys ropecalc does not recognize at all. */
  unknownKeys: string[];
}

/** Everything RoPE math needs to know about a model, from config.json or flags. */
export interface RopeGeometry {
  /** Where the geometry came from: a file path or "flags". */
  source: string;
  /** Full attention head dimension. */
  headDim: number;
  /** Rotary subspace dimension: `floor(headDim × partial_rotary_factor)`, always even. */
  rotaryDim: number;
  /** Number of rotated dimension pairs: `rotaryDim / 2`. */
  pairs: number;
  /** RoPE base (θ, config key `rope_theta`). */
  base: number;
  /** Context length the RoPE frequencies were trained at. */
  trainedCtx: number;
  /** `max_position_embeddings` as declared in the config, if any. */
  declaredMax: number | null;
  /** The config's `rope_scaling` block, if any. */
  scaling: RopeScalingBlock | null;
  /** Non-fatal normalization notes (derived head_dim, defaulted base, …). */
  warnings: string[];
}

/** One method's computed parameters inside a plan. */
export interface LinearPlan {
  factor: number;
  notes: string[];
}

export interface NtkPlan {
  /** The exponent `rotaryDim / (rotaryDim - 2)` applied to the factor. */
  exponent: number;
  /** The scaled base: `base × factor^exponent`. */
  scaledBase: number;
  notes: string[];
}

export interface YarnPlan {
  factor: number;
  betaFast: number;
  betaSlow: number;
  /** Correction-range boundaries in pair-index space (HF floor/ceil semantics). */
  low: number;
  high: number;
  /** Pair counts per zone: kept as trained / blended / fully interpolated. */
  zones: ZoneCounts;
  /** Attention temperature √(1/t) = 0.1·ln(factor) + 1. */
  mscale: number;
  notes: string[];
}

export interface ZoneCounts {
  keep: number;
  blend: number;
  interpolate: number;
}

export type MethodId = "linear" | "ntk" | "yarn";

export interface Recommendation {
  method: MethodId | "none";
  reason: string;
}

/** The result of `ropecalc plan`. */
export interface Plan {
  tool: "ropecalc";
  source: string;
  headDim: number;
  rotaryDim: number;
  base: number;
  trainedCtx: number;
  targetCtx: number;
  /** targetCtx / trainedCtx; ≤ 1 means no scaling is needed. */
  factor: number;
  finetune: boolean;
  linear: LinearPlan | null;
  ntk: NtkPlan | null;
  yarn: YarnPlan | null;
  recommendation: Recommendation;
  warnings: string[];
}

/** Severity of one `check` finding. `ok` findings record what passed. */
export type Level = "ok" | "info" | "warn" | "error";

export interface Finding {
  level: Level;
  message: string;
}

/** The result of `ropecalc check`. */
export interface CheckResult {
  tool: "ropecalc";
  source: string;
  scalingType: string | null;
  trainedCtx: number;
  declaredMax: number | null;
  /** trainedCtx × factor when a scaling block declares one, else trainedCtx. */
  effectiveMax: number;
  findings: Finding[];
  errors: number;
  warnings: number;
  valid: boolean;
}

/** Per-dimension-pair zone under YaRN blending. */
export type Zone = "keep" | "blend" | "interp";

export interface DimRow {
  pair: number;
  invFreq: number;
  /** Tokens per full rotation: 2π / invFreq. */
  wavelength: number;
  /** Full rotations completed over the trained context. */
  rotationsAtTrained: number;
  zone: Zone;
  /** Frequency divisors at the target factor: original invFreq / scaled invFreq. */
  linearDiv: number;
  ntkDiv: number;
  yarnDiv: number;
}

/** The result of `ropecalc dims`. */
export interface DimsResult {
  tool: "ropecalc";
  source: string;
  rotaryDim: number;
  pairs: number;
  base: number;
  trainedCtx: number;
  targetCtx: number;
  factor: number;
  betaFast: number;
  betaSlow: number;
  low: number;
  high: number;
  mscale: number;
  rows: DimRow[];
}
