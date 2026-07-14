/**
 * The `plan` engine: given a geometry and a target context, compute the
 * exact parameters every plannable method needs, plus one recommendation
 * with its reasoning stated — the whole point is to replace "someone on a
 * forum said factor 8" with numbers whose derivation is printed.
 */

import { ntkBase } from "./rope.js";
import {
  correctionRange,
  DEFAULT_BETA_FAST,
  DEFAULT_BETA_SLOW,
  yarnMscale,
  yarnZones,
} from "./yarn.js";
import { fmt2, fmtCtx, roundTo } from "./units.js";
import type { Plan, Recommendation, RopeGeometry } from "./types.js";

export interface PlanOptions {
  targetCtx: number;
  /** Whether the user intends to fine-tune at the extended length. */
  finetune?: boolean;
  betaFast?: number;
  betaSlow?: number;
}

/** Above this factor no published method has good results; the plan says so. */
const UNCHARTED_FACTOR = 32;

function recommend(factor: number, finetune: boolean): Recommendation {
  if (factor <= 1) {
    return {
      method: "none",
      reason: "the target is within the trained context; running unscaled is strictly better",
    };
  }
  if (finetune) {
    if (factor <= 4) {
      return {
        method: "linear",
        reason: `with fine-tuning at ${fmt2(factor)}×, plain position interpolation is the simplest recipe and every runtime supports it`,
      };
    }
    return {
      method: "yarn",
      reason: `at ${fmt2(factor)}× even fine-tuned linear interpolation loses high-frequency resolution; YaRN preserves it and fine-tunes with far less data`,
    };
  }
  if (factor <= 2) {
    return {
      method: "ntk",
      reason: `${fmt2(factor)}× without fine-tuning: NTK-by-base is a drop-in rope_theta override that keeps high-frequency (local) detail intact`,
    };
  }
  return {
    method: "yarn",
    reason: `${fmt2(factor)}× without fine-tuning is beyond what uniform tricks hold; YaRN's wavelength-aware blend plus attention temperature degrades the least`,
  };
}

/** Compute the full plan. Pure; rendering and emission live elsewhere. */
export function computePlan(geom: RopeGeometry, opts: PlanOptions): Plan {
  const finetune = opts.finetune === true;
  const factor = opts.targetCtx / geom.trainedCtx;
  const warnings: string[] = [];

  if (geom.scaling !== null && geom.scaling.type !== "" && geom.scaling.type !== "default") {
    warnings.push(
      `config already declares ${geom.scaling.type} scaling — the plan is computed from its original context ${fmtCtx(geom.trainedCtx)}`,
    );
  }
  if (factor > UNCHARTED_FACTOR) {
    warnings.push(
      `${fmt2(factor)}× is beyond published results for every method — validate on your own long-context tasks before trusting it`,
    );
  }

  const base: Plan = {
    tool: "ropecalc",
    source: geom.source,
    headDim: geom.headDim,
    rotaryDim: geom.rotaryDim,
    base: geom.base,
    trainedCtx: geom.trainedCtx,
    targetCtx: opts.targetCtx,
    factor: roundTo(factor, 4),
    finetune,
    linear: null,
    ntk: null,
    yarn: null,
    recommendation: recommend(factor, finetune),
    warnings,
  };

  if (factor <= 1) return base;

  base.linear = {
    factor: roundTo(factor, 4),
    notes: [
      `every pair's frequency divided by ${fmt2(factor)} — positions land ${fmt2(factor)}× closer together`,
      factor > 2
        ? `expect degraded retrieval without fine-tuning at ${fmt2(factor)}×`
        : `usable without fine-tuning at this modest factor`,
    ],
  };

  const exponent = geom.rotaryDim / (geom.rotaryDim - 2);
  base.ntk = {
    exponent: roundTo(exponent, 6),
    scaledBase: roundTo(ntkBase(geom.base, factor, geom.rotaryDim), 1),
    notes: [
      `fastest pair untouched, slowest compressed exactly ${fmt2(factor)}× — the spread is geometric in between`,
      `effective reach shrinks near the target; plan with headroom (target a larger context than you serve)`,
    ],
  };

  const betaFast = opts.betaFast ?? DEFAULT_BETA_FAST;
  const betaSlow = opts.betaSlow ?? DEFAULT_BETA_SLOW;
  const { low, high } = correctionRange(betaFast, betaSlow, geom.rotaryDim, geom.base, geom.trainedCtx);
  base.yarn = {
    factor: roundTo(factor, 4),
    betaFast,
    betaSlow,
    low,
    high,
    zones: yarnZones(low, high, geom.pairs),
    mscale: roundTo(yarnMscale(factor), 4),
    notes: [
      `pairs rotating ≥${betaFast}× over ${fmtCtx(geom.trainedCtx)} keep their trained frequency; pairs under ${betaSlow}× interpolate fully`,
      `attention logits scaled by ${fmt2(yarnMscale(factor))} (= 0.1·ln(${fmt2(factor)}) + 1)`,
    ],
  };

  return base;
}
