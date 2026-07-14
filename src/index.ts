/**
 * ropecalc public API — everything the CLI computes, importable as a library.
 * All functions are pure: geometry + options in, plain data out.
 */

export { VERSION } from "./version.js";
export {
  ConfigError,
  DEFAULT_BASE,
  geometryFromFlags,
  loadConfig,
  normalizeConfig,
  parseScalingBlock,
} from "./config.js";
export {
  dynamicNtkBase,
  invFreqs,
  ntkBase,
  ntkDivisor,
  rotations,
  TWO_PI,
  wavelength,
} from "./rope.js";
export {
  correctionDim,
  correctionRange,
  DEFAULT_BETA_FAST,
  DEFAULT_BETA_SLOW,
  extrapolationMask,
  yarnInvFreqs,
  yarnMscale,
  yarnZones,
} from "./yarn.js";
export {
  KNOWN_SCALING_TYPES,
  llama3InvFreqs,
  llama3Zones,
  methodById,
  METHODS,
  type MethodInfo,
} from "./methods.js";
export { checkConfig, type CheckOptions } from "./validate.js";
export { computePlan, type PlanOptions } from "./plan.js";
export { computeDims, ELIDE_THRESHOLD, selectRows, type DimsOptions, type RowPick } from "./dims.js";
export { emit, EmitError, emitHf, emitLlamaCpp, emitNotes, emitVllm, RUNTIMES, type Runtime } from "./emit.js";
export { renderCheck, renderDims, renderMethods, renderPlan } from "./render.js";
export { fmtCtx, fmtCtxShort, parseCtx, UnitError } from "./units.js";
export type {
  CheckResult,
  DimRow,
  DimsResult,
  Finding,
  Level,
  LinearPlan,
  MethodId,
  NtkPlan,
  Plan,
  Recommendation,
  RopeGeometry,
  RopeScalingBlock,
  YarnPlan,
  Zone,
  ZoneCounts,
} from "./types.js";
